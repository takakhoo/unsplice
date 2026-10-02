"""Closed-form recovery of speech features from the gradient of the first layer.

Any first layer that applies one affine map to overlapping patches of the feature sequence
(spliced context windows in DeepSpeech-1, a strided Conv1d in Transformer and Conformer
encoders) has gradient

    dL/dW = sum_p g_p patch_p^T,      dL/db = sum_p g_p,

so the rows of [dW | db] span the vectors [patch_p, 1]. Neighbouring patches share frames,
which turns "every [patch_p(X), 1] lies in that span" into a linear system in the unknown
features X. No transcript, loss, later layer, or optimisation is involved.

With F features per frame, kernel k and stride s, the system has a unique solution while
the number of patches P satisfies P <= (k - s) * F.
"""
import torch


def estimate_rank(s, floor=1e-6, min_gap=0.5, noise=1e-9, artifact_rule=True):
    """Number of independent patches = position of the largest drop in the singular values.

    Drops that land below `noise` are ignored: the last singular values of a float32
    gradient fall off a cliff of their own that has nothing to do with the signal rank.
    A float32 convolution backward pass can also leave one stray direction halfway between
    signal and noise; when the drop into it is itself an order of magnitude, the count stops there.
    """
    s = (s / s[0]).clamp_min(1e-300)
    logs = torch.log10(s)
    gaps = logs[:-1] - logs[1:]
    gaps = torch.where(s[1:] > noise, gaps, torch.zeros_like(gaps))
    k = int(torch.argmax(gaps)) + 1
    if gaps[k - 1] <= min_gap:
        return int((s > floor).sum())
    if artifact_rule and k >= 2 and gaps[k - 2] > 1.0:
        k -= 1
    return k


def patch_frame_index(T, kernel, stride, pad_left, n_patches=None):
    """(P, kernel) index of the frame each patch slot reads; -1 marks zero padding."""
    if n_patches is None:
        n_patches = (T + 2 * pad_left - kernel) // stride + 1
    idx = torch.arange(n_patches).unsqueeze(1) * stride - pad_left + torch.arange(kernel).unsqueeze(0)
    idx[(idx < 0) | (idx >= T)] = -1
    return idx


def extract_patches(X, kernel, stride, pad_left, n_patches=None):
    """X: (T, F) -> (P, kernel * F), slot j of patch p holding frame p*stride - pad_left + j."""
    T, F = X.shape
    idx = patch_frame_index(T, kernel, stride, pad_left, n_patches).to(X.device)
    Xz = torch.cat([X, X.new_zeros(1, F)], dim=0)
    return Xz[idx].reshape(idx.shape[0], kernel * F)


def _as_lengths(T):
    return [int(T)] if isinstance(T, int) else [int(t) for t in T]


def n_patches_for(T, kernel, stride, pad_left):
    return (T + 2 * pad_left - kernel) // stride + 1


def solve_patches(basis, n_feat, T, kernel, stride, pad_left, n_patches=None, ridge=0.0, has_bias=True,
                  check_unique=False):
    """Find X (sum(T), F) whose augmented patches all lie in span(basis).

    T is one length or a list of lengths (several utterances in the same update, each with
    its own zero padding). basis: (r, kernel * F + 1) orthonormal rows. Each patch touches a
    contiguous run of unknowns, so the normal matrix is a sum of overlapping kF x kF blocks
    along the diagonal. It is assembled in banded storage and solved by banded Cholesky.
    """
    import numpy as np
    from scipy.linalg import solveh_banded
    F, K = n_feat, kernel
    D = F * K
    lengths = _as_lengths(T)
    Pc = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device) - basis.T @ basis
    Q = Pc[:D, :D].cpu().numpy()
    q = Pc[:D, D].cpu().numpy() if has_bias else np.zeros(D)
    n, bw = sum(lengths) * F, D - 1
    band = np.zeros((bw + 1, n))
    rhs = np.zeros(n)
    diags_full = [np.diagonal(Q, -d).copy() for d in range(D)]
    start = 0
    for Tu in lengths:
        P = n_patches if (n_patches is not None and len(lengths) == 1) else n_patches_for(Tu, K, stride, pad_left)
        for p in range(P):
            first = p * stride - pad_left                   # frame held by slot 0
            lo, hi = max(0, -first), min(K, Tu - first)     # valid slots
            if hi <= lo:
                continue
            a, m = (start + first + lo) * F, (hi - lo) * F  # global offset and size of the block
            s0 = lo * F
            if lo == 0 and hi == K:
                for d in range(m):
                    band[d, a:a + m - d] += diags_full[d]
            else:
                Qs = Q[s0:s0 + m, s0:s0 + m]
                for d in range(m):
                    band[d, a:a + m - d] += np.diagonal(Qs, -d)
            rhs[a:a + m] -= q[s0:s0 + m]
        start += Tu
    if ridge:
        band[0] += ridge
    if not has_bias:
        raise NotImplementedError("layers without a bias leave the scale undetermined")
    x, bump = None, 0.0
    while x is None:
        try:
            x = solveh_banded(band, rhs, lower=True)
        except np.linalg.LinAlgError:   # rank-deficient system (over capacity): regularise minimally
            step = max(bump * 9, 1e-12 * band[0].max())
            band[0] += step
            bump += step
    ambiguity = None
    if check_unique:
        # Pull the solution towards a random point with a vanishing weight. A unique solution
        # does not move; a solution set with a free direction slides along it.
        lam = 1e-9 * band[0].max()
        target = np.random.default_rng(0).standard_normal(n) * (np.abs(x).mean() + 1e-12)
        band2 = band.copy()
        band2[0] += lam
        x2 = solveh_banded(band2, rhs + lam * target, lower=True)
        ambiguity = float(np.linalg.norm(x2 - x) / (np.linalg.norm(x) + 1e-30))
    return torch.from_numpy(x).reshape(sum(lengths), F).to(basis.device), ambiguity


def span_residual(X, basis, kernel, stride, pad_left, n_patches=None, has_bias=True, lengths=None):
    """Relative distance of each augmented patch of X from the span."""
    lengths = lengths or [X.shape[0]]
    parts, start = [], 0
    for Tu in lengths:
        parts.append(extract_patches(X[start:start + Tu], kernel, stride, pad_left,
                                     n_patches if len(lengths) == 1 else None))
        start += Tu
    p = torch.cat(parts)
    if has_bias:
        p = torch.cat([p, p.new_ones(len(p), 1)], dim=1)
    return (p - (p @ basis.T) @ basis).norm(dim=1) / p.norm(dim=1).clamp_min(1e-30)


def gradient_rowspace(dW, db=None):
    """Singular values and right singular vectors of [dW | db] in float64."""
    M = dW.reshape(dW.shape[0], -1).double()
    if db is not None:
        M = torch.cat([M, db.double().unsqueeze(1)], dim=1)
    _, s, vh = torch.linalg.svd(M, full_matrices=False)
    return s, vh


def linear_span_attack(dW, db, n_feat, kernel, stride=1, pad_left=0, T=None, n_patches=None,
                       rank=None, ridge=0.0):
    """Recover X (T, F) from one layer's gradient.

    dW: (H, kernel * F), slot-major (slot j occupies columns j*F:(j+1)*F). db: (H,) or None.
    The patch count is read off the rank. T may be given, or a list of candidates (each an
    int, or a tuple of lengths when several utterances share the update); the candidate whose
    solution sits closest to the span is returned.
    """
    s, vh = gradient_rowspace(dW, db)
    if rank is None:
        rank = estimate_rank(s)
    basis = vh[:rank]
    if n_patches is None:
        n_patches = rank
    if T is None:
        T = [n_patches * stride + kernel - stride - 2 * pad_left]
    cands = T if isinstance(T, (list, tuple)) else [T]
    tried = []
    for Tc in cands:
        multi = isinstance(Tc, (list, tuple))
        X, _ = solve_patches(basis, n_feat, Tc, kernel, stride, pad_left, None, ridge, has_bias=db is not None)
        res = float(span_residual(X, basis, kernel, stride, pad_left, None, has_bias=db is not None,
                                  lengths=list(Tc) if multi else None).max())
        tried.append((X, res, Tc))
    # Too short a length cannot fit the span; too long a one fits with spare frames. Take the
    # shortest length that fits about as well as the best.
    floor = min(r for _, r, _ in tried)
    best = next(t for t in tried if t[1] <= 3 * floor)
    return best[0], {"T": best[2], "n_patches": n_patches, "rank": rank, "singular_values": s,
                     "basis": basis, "max_residual": best[1], "candidates": [(t[2], t[1]) for t in tried]}


def shortest_fit(basis, n_feat, kernel, stride, pad_left, start, max_extra=48, patience=3, drop=3.0):
    """Shortest length whose solution lies in the span.

    The rank counts independent patches. It equals the patch count unless frames repeat
    exactly (digital silence), in which case the true length is longer. Lengths are tried in
    increasing order from the one implied by the rank; a length becomes the answer when it
    cuts the residual by a factor `drop`, and the search stops after `patience` lengths
    without such a cut.
    """
    best, tried, since = None, [], 0
    for T in range(start, start + max_extra + 1):
        X, _ = solve_patches(basis, n_feat, T, kernel, stride, pad_left)
        res = float(span_residual(X, basis, kernel, stride, pad_left).max())
        tried.append((T, res))
        if best is None or res * drop < best[2]:
            best, since = (X, T, res), 0
        else:
            since += 1
            if since >= patience:
                break
    _, ambiguity = solve_patches(basis, n_feat, best[1], kernel, stride, pad_left, check_unique=True)
    return best[0], best[1], best[2], tried, ambiguity


def ds1_attack(grads, n_feat=26, n_context=6, T=None, **kw):
    """DeepSpeech-1: fc1 acts on spliced windows, i.e. kernel 2c+1, stride 1, padding c."""
    s, vh = gradient_rowspace(grads["fc1.weight"], grads["fc1.bias"])
    rank = estimate_rank(s) if T is None else min(T, vh.shape[0] - 1)   # a known length fixes the span dimension
    basis = vh[:rank]
    K = 2 * n_context + 1
    if T is not None:
        X, _ = solve_patches(basis, n_feat, T, K, 1, n_context)
        X, amb = solve_patches(basis, n_feat, T, K, 1, n_context, check_unique=True)
        res, tried = float(span_residual(X, basis, K, 1, n_context).max()), [(T, None)]
    else:
        X, T, res, tried, amb = shortest_fit(basis, n_feat, K, 1, n_context, rank)
    return X, {"T": T, "rank": rank, "singular_values": s, "basis": basis, "max_residual": res,
               "candidates": tried, "ambiguity": amb}


def conv1d_attack(dW, db, n_feat, kernel, stride, pad, T=None):
    """Strided Conv1d front end. dW: (C_out, F, k) weight gradient, db: (C_out,)."""
    s, vh = gradient_rowspace(conv1d_weight_to_slot_major(dW), db)
    if T is not None:
        rank = min(n_patches_for(T, kernel, stride, pad), vh.shape[0] - 1)
        basis = vh[:rank]
        X, amb = solve_patches(basis, n_feat, T, kernel, stride, pad, check_unique=True)
        res = float(span_residual(X, basis, kernel, stride, pad).max())
        return X, {"T": T, "rank": rank, "singular_values": s, "basis": basis, "max_residual": res,
                   "candidates": [(T, res)], "ambiguity": amb}
    # A float32 convolution backward pass leaves one stray direction between signal and noise.
    # On trained weights it can sit close to the weakest real direction, so both readings of the
    # spectrum are fitted and the smaller rank is kept whenever it fits about as well.
    k = estimate_rank(s, artifact_rule=False)
    fits = []
    for rank in (k - 1, k):
        basis = vh[:rank]
        start = max(1, (rank - 1) * stride + kernel - 2 * pad)   # fewest frames that give `rank` patches
        X, T_fit, res, tried, amb = shortest_fit(basis, n_feat, kernel, stride, pad, start, patience=stride + 2, drop=10.0)
        fits.append((X, T_fit, res, tried, amb, rank, basis))
    best = fits[0] if fits[0][2] <= 10 * fits[1][2] else fits[1]
    X, T_fit, res, tried, amb, rank, basis = best
    return X, {"T": T_fit, "rank": rank, "singular_values": s, "basis": basis, "max_residual": res,
               "candidates": tried, "ambiguity": amb}


def conv1d_weight_to_slot_major(dW):
    """Conv1d weight gradient (C_out, F, k) -> (C_out, k * F) with slot-major columns."""
    return dW.permute(0, 2, 1).reshape(dW.shape[0], -1)
