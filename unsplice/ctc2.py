"""Length-aware log-space CTC with differentiable first and second derivatives.

Carried over from the earlier gradient-matching code base. Only the optimisation baseline
needs it: matching gradients by gradient descent differentiates the CTC loss twice, which
torch.nn.functional.ctc_loss does not support. The span attacks never call this.
Unreachable states are masked before logsumexp to avoid NaN double gradients.
"""
import torch


def _safe_logsumexp(values):
    reachable = torch.isfinite(values).any(dim=0)
    safe = torch.where(reachable.unsqueeze(0), values, torch.zeros_like(values))
    result = torch.logsumexp(safe, dim=0)
    return torch.where(reachable, result, torch.full_like(result, -torch.inf))


def ctc_loss_imp(log_probs, targets, input_lengths, target_lengths, blank=0,
                 reduction="mean", zero_infinity=False):
    if log_probs.ndim != 3 or not log_probs.is_floating_point():
        raise ValueError("log_probs must be floating (time, batch, classes)")
    time, batch, classes = log_probs.shape
    if batch < 1 or not 0 <= blank < classes or reduction not in ("none", "mean", "sum"):
        raise ValueError("invalid batch, blank or reduction")
    lengths = torch.as_tensor(input_lengths, dtype=torch.long).cpu()
    sizes = torch.as_tensor(target_lengths, dtype=torch.long).cpu()
    if lengths.shape != (batch,) or sizes.shape != (batch,) or torch.any(lengths < 0) or torch.any(lengths > time) or torch.any(sizes < 0):
        raise ValueError("invalid sequence lengths")
    targets = torch.as_tensor(targets, dtype=torch.long, device=log_probs.device)
    if targets.ndim == 1 and targets.numel() != int(sizes.sum()):
        raise ValueError("concatenated targets must match target_lengths")
    if targets.ndim == 2 and (targets.shape[0] != batch or targets.shape[1] < int(sizes.max())):
        raise ValueError("padded target shape does not match target_lengths")
    if targets.ndim not in (1, 2):
        raise ValueError("targets must be concatenated or padded")
    losses, offset = [], 0
    for b, (length, size) in enumerate(zip(lengths.tolist(), sizes.tolist())):
        labels = targets[b, :size] if targets.ndim == 2 else targets[offset:offset+size]
        offset += size
        if torch.any(labels < 0) or torch.any(labels >= classes) or torch.any(labels == blank):
            raise ValueError("target labels must be valid nonblank class indices")
        lp = log_probs[:length, b].double()
        if not torch.isfinite(lp).all():
            raise ValueError("finite log probabilities are required in valid frames")
        minimum = size + int((labels[1:] == labels[:-1]).sum())
        if length < minimum:
            # No alignment exists. Gradients of infinity are undefined; when
            # zero_infinity is enabled this branch is connected with zero grad.
            losses.append(lp.sum()*0 + (0 if zero_infinity else torch.inf))
            continue
        if length == 0:
            losses.append(lp.sum()*0)
            continue
        states = labels.new_full((2*size+1,), blank)
        states[1::2] = labels
        reachable = torch.arange(len(states), device=lp.device) < min(2, len(states))
        alpha = torch.where(reachable, lp[0, states], lp.new_full((len(states),), -torch.inf))
        skip_allowed = torch.zeros(len(states), dtype=torch.bool, device=lp.device)
        skip_allowed[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
        for t in range(1, length):
            prev = torch.cat((alpha.new_full((1,), -torch.inf), alpha[:-1]))
            skip = torch.cat((alpha.new_full((min(2, len(states)),), -torch.inf), alpha[:-2]))
            skip = torch.where(skip_allowed, skip, torch.full_like(skip, -torch.inf))
            alpha = _safe_logsumexp(torch.stack((alpha, prev, skip))) + lp[t, states]
        losses.append(-_safe_logsumexp(alpha[-2:].reshape(-1, 1))[0])
    result = torch.stack(losses).to(log_probs.dtype)
    if reduction == "sum":
        return result.sum()
    if reduction == "mean":
        return (result / sizes.clamp_min(1).to(result.device)).mean()
    return result
