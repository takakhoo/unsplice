"""Long-form recovery from the spans leaked by deeper per-frame layers.

Layer l (fc2, fc3, LSTM input weights) has gradient sum_t g_t h_{l-1,t}^T, so its rows span
the layer inputs [h_{l-1,t}, 1]. Each h is a known nonlinear function of one context window,
which lets the attacker test any candidate frame against the span without touching the
LSTM, the loss, or the transcript. The objective below is that test, summed over frames.
"""
import torch
from .model import splice


def aug_rowspace(weight_grad, bias_grad, rank=None, floor=1e-6, seed=0):
    """Singular values and right singular vectors of [dW | db].

    A tall matrix is first multiplied on the left by a Gaussian sketch with a few more rows
    than it has columns. That keeps the row space and makes the SVD several times cheaper.
    """
    M = torch.cat([weight_grad, bias_grad.unsqueeze(1)], dim=1).double()
    rows, cols = M.shape
    if rows > cols + 256:
        g = torch.Generator(device=M.device).manual_seed(seed)
        sketch = torch.randn(cols + 128, rows, generator=g, dtype=M.dtype, device=M.device) / (cols + 128) ** 0.5
        M = sketch @ M
    _, s, vh = torch.linalg.svd(M, full_matrices=False)
    return s, vh


def gap_rank(s, floor=1e-6, min_gap=0.5, noise=1e-9):
    """Frame count = position of the largest drop in the singular values.

    Drops that land below `noise` are ignored: the last few singular values of a float32
    gradient fall off a cliff of their own that has nothing to do with the signal rank.
    """
    s = (s / s[0]).clamp_min(1e-300)
    logs = torch.log10(s)
    gaps = logs[:-1] - logs[1:]
    gaps = torch.where(s[1:] > noise, gaps, torch.zeros_like(gaps))
    k = int(torch.argmax(gaps)) + 1
    return k if gaps[k - 1] > min_gap else int((s > floor).sum())


def leaked_spans(grads, layers=("fc2", "fc3", "lstm"), T=None):
    """Orthonormal bases of span{[h_{l-1,t}, 1]} for the chosen layers, and the frame count."""
    raw = {}
    if "fc1" in layers:
        raw["fc1"] = aug_rowspace(grads["fc1.weight"], grads["fc1.bias"])
    if "fc2" in layers:
        raw["fc2"] = aug_rowspace(grads["fc2.weight"], grads["fc2.bias"])
    if "fc3" in layers:
        raw["fc3"] = aug_rowspace(grads["fc3.weight"], grads["fc3.bias"])
    if "lstm" in layers:
        w = torch.cat([grads["lstm.weight_ih_l0"], grads["lstm.weight_ih_l0_reverse"]], dim=0)
        b = torch.cat([grads["lstm.bias_ih_l0"], grads["lstm.bias_ih_l0_reverse"]], dim=0)
        raw["lstm"] = aug_rowspace(w, b)
    if T is None:
        # Take the count from the layer with the cleanest gap; a saturated layer reports its width.
        est = {k: gap_rank(s) for k, (s, vh) in raw.items()}
        T = max(v for k, v in est.items() if v < raw[k][1].shape[1] - 1) if any(
            v < raw[k][1].shape[1] - 1 for k, v in est.items()) else max(est.values())
    bases = {k: vh[:min(T, vh.shape[0] - 1)] for k, (s, vh) in raw.items()}
    svals = {k: s for k, (s, vh) in raw.items()}
    return bases, T, svals


LAYER_INPUT = {"fc1": 0, "fc2": 1, "fc3": 2, "lstm": 3}


def span_objective(model, X, bases, reduce=True):
    """Sum over frames and layers of the squared distance from [h, 1] to the leaked span."""
    feats = model.front(X.unsqueeze(1), upto=max(LAYER_INPUT[k] for k in bases))
    total, per_layer = 0.0, {}
    for name, B in bases.items():
        h = feats[LAYER_INPUT[name]].squeeze(1).to(B.dtype)
        v = torch.cat([h, torch.ones(len(h), 1, dtype=h.dtype, device=h.device)], dim=1)
        r = v - (v @ B.T) @ B
        e = (r * r).sum(1) / (v * v).sum(1)
        per_layer[name] = e
        total = total + e
    return (total.sum() if reduce else total), per_layer


def deep_span_attack(model, grads, layers=("fc2", "fc3", "lstm"), T=None, steps=3000, lr=0.3,
                     init=None, x_true=None, log_every=0, lbfgs_steps=0, seed=0):
    """Recover X (T, F) by driving every frame's lifted features into the leaked spans."""
    dev = next(model.parameters()).device
    bases, T, svals = leaked_spans(grads, layers, T)
    dmodel = _double_front(model)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if init is None:
        X = torch.randn(T, model.n_feat, generator=g, dtype=torch.float64).to(dev) * 0.1
    else:
        X = init.clone().double().to(dev)
    X.requires_grad_(True)
    opt = torch.optim.Adam([X], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    hist = []
    for i in range(steps):
        opt.zero_grad()
        loss, _ = span_objective(dmodel, X, bases)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (i % log_every == 0 or i == steps - 1):
            rec = {"step": i, "loss": float(loss)}
            if x_true is not None:
                rec["mae"] = float((X.detach() - x_true.double()).abs().mean())
            hist.append(rec)
            print(rec, flush=True)
    if lbfgs_steps:
        opt2 = torch.optim.LBFGS([X], lr=1.0, max_iter=lbfgs_steps, history_size=50,
                                 tolerance_grad=1e-14, tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss, _ = span_objective(dmodel, X, bases)
            loss.backward()
            return loss
        opt2.step(closure)
    with torch.no_grad():
        final, per_layer = span_objective(dmodel, X, bases, reduce=False)
    return X.detach(), {"T": T, "frame_residual": final, "history": hist, "singular_values": svals}


class _double_front:
    """Float64 copy of the per-frame front end (fc1..fc3) used by the objective."""

    def __init__(self, model):
        self.n_context, self.relu, self.relu_clip = model.n_context, model.relu, model.relu_clip
        self.layers = [(fc.weight.detach().double(), fc.bias.detach().double())
                       for fc in (model.fc1, model.fc2, model.fc3)]

    def act(self, a):
        return torch.clamp(a, 0.0, self.relu_clip) if self.relu else a

    def front(self, x, upto=3):
        feats = [splice(x, self.n_context)]
        for w, b in self.layers[:upto]:
            feats.append(self.act(feats[-1] @ w.T + b))
        return feats
