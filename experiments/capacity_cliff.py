"""Error of the closed-form attack as the number of patches crosses the predicted limit (k - s) * F.

Real LibriSpeech features are cropped to an exact length, a random-weight front end plus a small
nonlinear network produces a float32 gradient, and the attack is given the length (the question
here is uniqueness of the linear system, not length estimation).

Usage: python experiments/capacity_cliff.py --feats DIR --out FILE.jsonl
"""
import argparse, json, sys
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.linear_attack import (gradient_rowspace, solve_patches, conv1d_weight_to_slot_major,
                                    extract_patches, n_patches_for)
from unsplice.data import FeatureSet
from unsplice.tfctc import normalise


def run_case(x, kernel, stride, pad, width, dev):
    """x: (T, F). Returns relative error of the reconstruction."""
    T, F = x.shape
    P = n_patches_for(T, kernel, stride, pad)
    layer = nn.Linear(kernel * F, width).to(dev)
    rest = nn.Sequential(nn.ReLU(), nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 29)).to(dev)
    patches = extract_patches(x, kernel, stride, pad)
    z = rest(layer(patches))
    y = torch.randint(1, 29, (max(2, P // 8),), device=dev)
    loss = nn.functional.ctc_loss(z.log_softmax(-1).unsqueeze(1), y.unsqueeze(0), torch.tensor([P]), torch.tensor([len(y)]),
                                  zero_infinity=True)
    gW, gb = torch.autograd.grad(loss, [layer.weight, layer.bias])
    s, vh = gradient_rowspace(gW, gb)
    rank = min(P, vh.shape[0] - 1)
    X, amb = solve_patches(vh[:rank], F, T, kernel, stride, pad, check_unique=True)
    err = X - x.double()
    return {"P": P, "T": T, "rel": float(err.norm() / x.double().norm()), "mae": float(err.abs().mean()), "ambiguity": amb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    mf = FeatureSet(a.feats, "test-clean", "mfcc")
    fb = FeatureSet(a.feats, "test-clean", "fbank")
    long_m = [i for i in range(len(mf)) if mf.frames(i) > 700]
    long_f = [i for i in range(len(fb)) if fb.frames(i) > 1400]
    configs = [
        {"name": "DeepSpeech-1, context 6 (k=13, s=1, F=26)", "kind": "mfcc", "k": 13, "s": 1, "pad": 6, "width": 2048},
        {"name": "DeepSpeech-1, context 9 (k=19, s=1, F=26)", "kind": "mfcc", "k": 19, "s": 1, "pad": 9, "width": 2048},
        {"name": "Transformer-CTC conv (k=7, s=3, F=80)", "kind": "fbank", "k": 7, "s": 3, "pad": 3, "width": 1536},
        {"name": "Conv (k=5, s=2, F=80)", "kind": "fbank", "k": 5, "s": 2, "pad": 2, "width": 1024},
    ]
    with open(a.out, "a") as f:
        for cfg in configs:
            F = 26 if cfg["kind"] == "mfcc" else 80
            cap = (cfg["k"] - cfg["s"]) * F
            grid = sorted(set([cap // 4, cap // 2, 3 * cap // 4] + list(range(cap - 24, cap - 4, 4)) + list(range(cap - 4, cap + 5))
                              + list(range(cap + 8, cap + 33, 8))))
            for P in grid:
                T = (P - 1) * cfg["s"] + cfg["k"] - 2 * cfg["pad"]
                for rep in range(a.reps):
                    if cfg["kind"] == "mfcc":
                        x = mf[long_m[rep % len(long_m)]][0][100:100 + T].to(dev)
                    else:
                        x = normalise(fb[long_f[rep % len(long_f)]][0][200:200 + T].unsqueeze(0)).squeeze(0).to(dev)
                    rec = run_case(x, cfg["k"], cfg["s"], cfg["pad"], cfg["width"], dev)
                    rec.update(config=cfg["name"], capacity=cap, rep=rep)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            print(cfg["name"], "done", flush=True)


if __name__ == "__main__":
    main()
