"""Sequential span decoding for utterances past the first-layer limit.

The spans leaked by fc2, fc3 and the LSTM input weights contain the lifted features of every
frame. A candidate context window can be tested against them, but the test alone does not
say which frame it is. The zero padding at the two ends of the utterance does: the first
window is [0, ..., 0, x_0, ..., x_c], and each later window adds exactly one new frame.
So the decoder solves the first window, then extends one frame at a time (F unknowns per
step), checking every step against the spans, and finishes with a joint polish.
"""
import torch
from torch.func import jacfwd
from .deep_attack import leaked_spans, LAYER_INPUT, _double_front, span_objective


class SpanDecoder:
    def __init__(self, model, grads, layers=("fc2", "fc3", "lstm"), T=None, codebook=None):
        """codebook: optional (N, F) frames of public speech, used to pick restart points."""
        self.codebook = None if codebook is None else codebook.double().to(next(model.parameters()).device)
        self.bases, self.T, self.svals = leaked_spans(grads, layers, T)
        self.front = _double_front(model)
        self.F, self.c = model.n_feat, model.n_context
        self.K = 2 * self.c + 1
        self.dev = next(model.parameters()).device
        self.upto = max(LAYER_INPUT[k] for k in self.bases)

    def residual(self, ctx):
        """ctx: (D,) one context window. Returns the stacked, scale-free span residual."""
        feats = [ctx]
        for w, b in self.front.layers[:self.upto]:
            feats.append(self.front.act(feats[-1] @ w.T + b))
        out = []
        for name, B in self.bases.items():
            h = feats[LAYER_INPUT[name]]
            v = torch.cat([h, h.new_ones(1)])
            r = v - (v @ B.T) @ B
            out.append(r / v.norm())
        return torch.cat(out)

    def residual_norms(self, ctx):
        """ctx: (N, D) candidate windows -> (N,) residual norms, all in one batch."""
        feats = [ctx]
        for w, b in self.front.layers[:self.upto]:
            feats.append(self.front.act(feats[-1] @ w.T + b))
        total = 0.0
        for name, B in self.bases.items():
            h = feats[LAYER_INPUT[name]]
            v = torch.cat([h, h.new_ones(len(h), 1)], dim=1)
            r = v - (v @ B.T) @ B
            total = total + (r * r).sum(1) / (v * v).sum(1)
        return total.sqrt()

    def lm(self, f, u0, iters=60, lam=1e-2, rtol=1e-3):
        """Levenberg-Marquardt on a small unknown vector u. Returns (u, residual norm).

        Stops when a step no longer shrinks the cost by a fraction rtol: with a float32
        gradient the residual at the true frame is a noise floor, not zero.
        """
        u = u0.clone()
        r = f(u)
        cost = float(r @ r)
        J_of = jacfwd(f)
        for _ in range(iters):
            J = J_of(u)
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

    def solve_first(self, inits=None, restarts=24, scale=5.0, seed=0, accept=1e-3, reverse=False):
        """First (or last) context window: c + 1 unknown frames next to c blocks of zero padding.

        inits: optional (N, c + 1, F) starting points, e.g. opening windows of public utterances.
        Random starts are used once those run out.
        """
        n = (self.c + 1) * self.F
        zeros = torch.zeros(self.c * self.F, dtype=torch.float64, device=self.dev)
        f = (lambda u: self.residual(torch.cat([u, zeros]))) if reverse else (lambda u: self.residual(torch.cat([zeros, u])))
        g = torch.Generator(device="cpu").manual_seed(seed)
        best, tried = None, 0
        n_prior = 0 if inits is None else len(inits)
        for k in range(max(restarts, n_prior)):
            if k < n_prior:
                u0 = inits[k].reshape(-1).double().to(self.dev)
            else:
                u0 = (torch.randn(n, generator=g, dtype=torch.float64) * scale).to(self.dev)
            u, res = self.lm(f, u0, iters=200)
            tried += 1
            if best is None or res < best[1]:
                best = (u, res)
            if res < accept:
                break
        self.first_tries = tried
        return best[0].reshape(self.c + 1, self.F), best[1]

    def extend(self, known, prev, reverse=False, accept=3e-3, restarts=3, top=6, seed=0):
        """One step: known is (K-1, F), the frames shared with the previous window.

        Tries the previous frame as the starting point (speech moves slowly), then the public
        frames that already sit closest to the span, then random perturbations.
        """
        flat = known.reshape(-1)
        f = (lambda x: self.residual(torch.cat([x, flat]))) if reverse else (lambda x: self.residual(torch.cat([flat, x])))
        best = self.lm(f, prev, iters=40)
        if best[1] < accept:
            return best
        starts = []
        if self.codebook is not None:
            cb = self.codebook
            rep = flat.unsqueeze(0).expand(len(cb), -1)
            ctx = torch.cat([cb, rep], dim=1) if reverse else torch.cat([rep, cb], dim=1)
            order = torch.argsort(self.residual_norms(ctx))[:top]
            starts += [cb[i] for i in order]
        g = torch.Generator(device="cpu").manual_seed(seed)
        starts += [prev + (torch.randn(self.F, generator=g, dtype=torch.float64) * 3.0 * k).to(self.dev) for k in range(1, restarts + 1)]
        for x0 in starts:
            x, res = self.lm(f, x0, iters=40)
            if res < best[1]:
                best = (x, res)
            if res < accept:
                break
        return best

    def refine_tail(self, frames, m=26, iters=25):
        """Jointly re-fit the last m decoded frames against every complete window they belong to.

        frames: list in decode order (so always "zero padding, then frames" with the newest last;
        a backward chain is the mirror image and uses the mirrored windows). Without this the
        small error of each step feeds the next one and the chain drifts off the span.
        """
        c, F = self.c, self.F
        n = len(frames)
        m = min(m, n)
        fixed = torch.stack(frames[:n - m]) if n > m else torch.zeros(0, F, dtype=torch.float64, device=self.dev)
        tail = torch.stack(frames[n - m:]).clone().requires_grad_(True)
        pad = torch.zeros(c, F, dtype=torch.float64, device=self.dev)
        lo = max(0, n - m - c)                         # first window (by centre) that touches the tail
        opt = torch.optim.LBFGS([tail], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

        def cost():
            seq = torch.cat([pad, fixed, tail])       # index j holds frame j - c
            wins = torch.stack([seq[t:t + 2 * c + 1] for t in range(lo, n - c)])     # centres lo .. n-c-1
            if self.reverse:
                wins = wins.flip(1)
            return self.residual_norms(wins.reshape(len(wins), -1)).pow(2).sum()

        def closure():
            opt.zero_grad()
            loss = cost()
            loss.backward()
            return loss
        if n - c > lo:
            opt.step(closure)
        return frames[:n - m] + list(tail.detach())

    def chain(self, inits=None, reverse=False, accept=3e-3, max_frames=None, verbose=False, refine_every=8):
        """Decode from one end until a window can no longer be placed in the span.

        Returns (frames, residuals, finished). Frames are in time order. `finished` means the
        chain ran into the zero padding at the far end, so `frames` is the whole utterance.
        The length is not assumed: real MFCC frames are never zero, and the c windows after
        the last frame each add an all-zero frame, which is how the end is recognised.
        """
        c, F = self.c, self.F
        max_frames = max_frames or (self.T + 4 * c + 64)
        first, r0 = self.solve_first(inits=inits, reverse=reverse, accept=1e-3)
        if r0 > 1e-3:
            return torch.zeros(0, F, dtype=torch.float64, device=self.dev), [r0], False
        frames = list(first.flip(0)) if reverse else list(first)      # stored in decode order
        self.reverse = reverse
        res, zeros_seen, finished = [r0], 0, False
        while len(frames) < max_frames:
            known = torch.stack(frames[-2 * c:])
            if len(known) < 2 * c:
                pad = torch.zeros(2 * c - len(known), F, dtype=torch.float64, device=self.dev)
                known = torch.cat([pad, known])
            if reverse:
                known = known.flip(0)
            x, r = self.extend(known, frames[-1], reverse=reverse, accept=accept)
            if r > accept:
                break
            frames.append(x)
            res.append(r)
            if refine_every and len(frames) % refine_every == 0 and zeros_seen == 0:
                frames = self.refine_tail(frames)
            zeros_seen = zeros_seen + 1 if float(x.abs().max()) < 1e-2 else 0
            if zeros_seen == c:
                finished = True
                frames = frames[:-c]
                break
            if verbose and len(frames) % 200 == 0:
                print(f"  {'backward' if reverse else 'forward'} chain at {len(frames)} frames, residual {r:.1e}", flush=True)
        X = torch.stack(frames)
        return (X.flip(0) if reverse else X), res, finished

    def decode(self, inits=None, last_inits=None, accept=3e-3, verbose=False):
        """Full utterance: forward chain, and a backward chain to cover whatever the forward one missed.

        Returns X (T, F) and a dict with the decoded length and how much each chain covered.
        """
        Xf, rf, done = self.chain(inits=inits, accept=accept, verbose=verbose)
        info = {"forward_frames": len(Xf), "forward_finished": done, "first_tries": getattr(self, "first_tries", 0)}
        if done:
            info["T"] = len(Xf)
            return Xf, info
        Xb, rb, done_b = self.chain(inits=last_inits, reverse=True, accept=accept, verbose=verbose)
        info.update(backward_frames=len(Xb), backward_finished=done_b)
        if done_b:
            info["T"] = len(Xb)
            return Xb, info
        T = self.T
        info["T"] = T
        X = torch.zeros(T, self.F, dtype=torch.float64, device=self.dev)
        nf, nb = min(len(Xf), T), min(len(Xb), T)
        X[:nf] = Xf[:nf]
        if nb:
            X[T - nb:] = Xb[len(Xb) - nb:]
        info["covered"] = min(1.0, (nf + nb) / T)
        return X, info

    def polish(self, X, steps=300):
        """Joint refinement of all frames against all spans, starting inside the right basin."""
        X = X.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([X], lr=1.0, max_iter=steps, history_size=30, tolerance_grad=1e-16,
                                tolerance_change=1e-18, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss, _ = span_objective(self.front, X, self.bases)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            per_frame, _ = span_objective(self.front, X, self.bases, reduce=False)
        return X.detach(), per_frame
