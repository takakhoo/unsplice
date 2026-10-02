"""Scoring a reconstruction against the true features."""
import torch


def feature_metrics(X, x):
    """X, x: (T, F) of the same length."""
    err = X.double() - x.double()
    return {"mae": float(err.abs().mean()), "rel": float(err.norm() / x.double().norm()),
            "max_frame_mae": float(err.abs().mean(1).max()),
            "snr_db": float(10 * torch.log10(x.double().pow(2).sum() / err.pow(2).sum().clamp_min(1e-300)))}


def align_shorter(X, x):
    """X has fewer frames than x. Find the monotone match of X into x with least total error.

    Used when exactly repeated frames (digital silence) make the length ambiguous: the
    reconstruction then equals the truth with some repeated frames dropped. Returns the
    indices of x matched to each frame of X.
    """
    n, m = X.shape[0], x.shape[0]
    cost = torch.cdist(X.double(), x.double(), p=1).cpu()      # (n, m)
    best = torch.full((n, m), float("inf"), dtype=torch.float64)
    best[0] = cost[0]
    for i in range(1, n):
        prev = torch.cummin(best[i - 1], dim=0).values       # best over j' < j
        best[i, 1:] = cost[i, 1:] + prev[:-1]
    j = int(torch.argmin(best[n - 1]))
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(torch.argmin(best[i - 1, :j]))
        path.append(j)
    return torch.tensor(path[::-1])


def score(X, x, max_gap=64):
    """Metrics with a length check. If X is a little shorter than x, score it after alignment
    and report how many frames are missing; any other mismatch scores as a blank reconstruction."""
    T, Te = x.shape[0], X.shape[0]
    if Te == T:
        return {**feature_metrics(X, x), "length_error": 0}
    if 0 < T - Te <= max_gap and Te > 0:
        idx = align_shorter(X, x)
        return {**feature_metrics(X, x[idx.to(x.device)]), "length_error": Te - T}
    return {**feature_metrics(torch.zeros_like(x, dtype=torch.float64), x), "length_error": Te - T}
