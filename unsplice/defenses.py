"""Transformations a client (or the protocol) applies to an update before the server sees it."""
import torch


def apply_defense(grads, spec):
    """spec examples: none | noise:1e-3 | clip:1.0 | dp:1.0:1e-2 | prune:0.9 | sign | quant:8 | f16

    noise:s    Gaussian noise with std s * (RMS of that tensor), per tensor.
    clip:c     scale the whole update to L2 norm at most c.
    dp:c:s     clip to norm c, then add Gaussian noise with std s * c to every coordinate.
    prune:p    zero the fraction p of smallest-magnitude entries in each tensor.
    sign       keep only the sign of each entry (signSGD).
    quant:b    uniform b-bit quantisation of each tensor between its min and max.
    f16        cast to half precision.
    """
    if spec in (None, "none", ""):
        return grads
    parts = spec.split(":")
    kind = parts[0]
    out = {}
    if kind == "noise":
        s = float(parts[1])
        for k, g in grads.items():
            out[k] = g + torch.randn_like(g) * s * g.pow(2).mean().sqrt()
    elif kind in ("clip", "dp"):
        c = float(parts[1])
        total = torch.sqrt(sum(g.pow(2).sum() for g in grads.values()))
        scale = min(1.0, c / float(total))
        for k, g in grads.items():
            out[k] = g * scale
            if kind == "dp":
                out[k] = out[k] + torch.randn_like(g) * float(parts[2]) * c
    elif kind == "prune":
        p = float(parts[1])
        for k, g in grads.items():
            if g.numel() < 2:
                out[k] = g
                continue
            thresh = torch.quantile(g.abs().flatten()[:: max(1, g.numel() // 1_000_000)].float(), p)
            out[k] = torch.where(g.abs() >= thresh, g, torch.zeros_like(g))
    elif kind == "sign":
        out = {k: torch.sign(g) for k, g in grads.items()}
    elif kind == "quant":
        levels = 2 ** int(parts[1]) - 1
        for k, g in grads.items():
            lo, hi = g.min(), g.max()
            out[k] = torch.round((g - lo) / (hi - lo).clamp_min(1e-30) * levels) / levels * (hi - lo) + lo
    elif kind == "f16":
        out = {k: g.half().float() for k, g in grads.items()}
    else:
        raise ValueError(spec)
    return out
