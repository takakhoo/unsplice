"""Span decoding for any front end whose features are local in time.

Setting: a nonlinear front end phi maps each window of k input frames (stride s) to a vector,
and the next layer is linear in that vector, so its gradient leaks span{[phi(window_p), 1]}.
A window can be tested against the span, which gives a nonlinear least-squares problem per
window. Once one window is found, its neighbours share k - s frames with it, so the utterance
is decoded outwards s frames at a time.

This covers Conformer / ESPnet style Conv2dSubsampling followed by a linear projection.
"""
import torch
from torch.func import jvp, vmap


def jacobian_fwd(f, u, chunk=64):
    """Forward-mode Jacobian (m, n) of f at u, a few input directions at a time to bound memory."""
    eye = torch.eye(u.numel(), dtype=u.dtype, device=u.device)
    cols = [vmap(lambda v: jvp(f, (u,), (v,))[1])(eye[i:i + chunk]) for i in range(0, u.numel(), chunk)]
    return torch.cat(cols).T


def lm_solve(f, u0, iters=80, lam=1e-2, rtol=1e-3, chunk=64):
    """Levenberg-Marquardt for a small residual function f(u). Returns (u, residual norm)."""
    u = u0.clone()
    r = f(u)
    cost = float(r @ r)
    for _ in range(iters):
        J = jacobian_fwd(f, u, chunk)
        g = J.T @ r
        H = J.T @ J
        damp = torch.diag(torch.diagonal(H).clamp_min(1e-12))
        improved = False
        for _ in range(8):
            step = torch.linalg.solve(H + lam * damp, -g)
            r_new = f(u + step)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                gain = (cost - cost_new) / cost
                u, r, cost, improved = u + step, r_new, cost_new, True
                lam = max(lam / 3, 1e-9)
                break
            lam *= 4
        if not improved or gain < rtol:
            break
    return u, cost ** 0.5


class LocalSpanDecoder:
    def __init__(self, phi, basis, k, s, n_feat, phi_all=None):
        """phi: (k, F) float64 window -> 1-D lifted vector. basis: (r, dim + 1) orthonormal rows.

        phi_all: optional, maps a stretch of frames (n, F) to the lifted vectors of all its
        windows at once; enables the joint refinement that keeps long chains on track.
        """
        self.phi, self.B, self.k, self.s, self.F, self.phi_all = phi, basis, k, s, n_feat, phi_all

    def polish(self, frames, iters=60):
        """Refine every decoded frame against every window it takes part in."""
        if self.phi_all is None:
            return frames, None
        X = frames.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([X], lr=1.0, max_iter=iters, history_size=20, tolerance_grad=1e-14,
                                tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def cost():
            h = self.phi_all(X)
            v = torch.cat([h, h.new_ones(len(h), 1)], dim=1)
            r = v - (v @ self.B.T) @ self.B
            return (r * r).sum(1) / (v * v).sum(1)

        def closure():
            opt.zero_grad()
            loss = cost().sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            worst = float(cost().max().sqrt())
        return X.detach(), worst

    def residual(self, window):
        h = self.phi(window)
        v = torch.cat([h, h.new_ones(1)])
        r = v - (v @ self.B.T) @ self.B
        return r / v.norm()

    def seed(self, inits, accept=1e-3):
        """Find any one window of the utterance. inits: (N, k, F) starting points."""
        best = None
        for n, w0 in enumerate(inits):
            u, res = lm_solve(lambda u: self.residual(u.reshape(self.k, self.F)), w0.reshape(-1).double(), iters=150)
            if best is None or res < best[1]:
                best = (u.reshape(self.k, self.F), res, n + 1)
            if res < accept:
                break
        return best

    def extend(self, frames, direction, accept=1e-3, max_steps=10 ** 6, restarts=4, polish_every=4):
        """Grow a decoded stretch by s frames per step until a window no longer fits."""
        k, s, F = self.k, self.s, self.F
        gen = torch.Generator(device="cpu").manual_seed(0)
        steps, retried = 0, False
        while steps < max_steps:
            if direction > 0:
                known = frames[-(k - s):]
                f = lambda u: self.residual(torch.cat([known, u.reshape(s, F)]))
                guess = frames[-1:].repeat(s, 1)
            else:
                known = frames[:k - s]
                f = lambda u: self.residual(torch.cat([u.reshape(s, F), known]))
                guess = frames[:1].repeat(s, 1)
            best = None
            for j in range(restarts):
                u0 = guess.reshape(-1) + (0 if j == 0 else torch.randn(s * F, generator=gen, dtype=torch.float64).to(frames.device) * 0.5 * j)
                u, res = lm_solve(f, u0)
                if best is None or res < best[1]:
                    best = (u, res)
                if res < accept:
                    break
            if best[1] > accept:
                if retried or self.phi_all is None:
                    break
                frames, _ = self.polish(frames)      # clean up accumulated error, then try this step once more
                retried = True
                continue
            retried = False
            new = best[0].reshape(s, F)
            frames = torch.cat([frames, new]) if direction > 0 else torch.cat([new, frames])
            steps += 1
            if polish_every and steps % polish_every == 0:
                frames, _ = self.polish(frames, iters=20)
        return frames, steps

    def decode(self, inits, accept=1e-3):
        w, res, tries = self.seed(inits, accept)
        if res > accept:
            return None, {"seed_residual": res, "seed_tries": tries}
        frames, right = self.extend(w, +1, accept)
        frames, left = self.extend(frames, -1, accept)
        frames, worst = self.polish(frames)
        return frames, {"seed_residual": res, "seed_tries": tries, "right_steps": right, "left_steps": left,
                        "final_residual": worst}

    def decode_segments(self, inits, accept=1e-3, max_segments=12):
        """Seed from every starting point and grow each new seed into a segment.

        Different starting points fall onto different windows of the utterance, so the union
        of the grown segments covers more than a single chain does. A seed that lands inside
        an existing segment is skipped. Returns the list of segments, each (n, F).
        """
        segments, seeds = [], 0
        for w0 in inits:
            u, res = lm_solve(lambda u: self.residual(u.reshape(self.k, self.F)), w0.reshape(-1).double(), iters=60)
            if res > accept:
                continue
            w = u.reshape(self.k, self.F)
            seeds += 1
            if any(_contains(seg, w) for seg in segments):
                continue
            frames, _ = self.extend(w, +1, accept)
            frames, _ = self.extend(frames, -1, accept)
            frames, _ = self.polish(frames)
            segments = [seg for seg in segments if not _contains(frames, seg[:self.k])]
            segments.append(frames)
            if len(segments) >= max_segments:
                break
        return segments, {"seeds": seeds}


def _contains(seg, w, tol=0.05):
    """True if window w (k, F) appears inside seg at some offset."""
    k = w.shape[0]
    if seg.shape[0] < k:
        return False
    windows = seg.unfold(0, k, 1).permute(0, 2, 1)             # (n - k + 1, k, F)
    return bool(((windows - w).abs().mean(dim=(1, 2)) < tol).any())
