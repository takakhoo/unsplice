"""When does additive noise hide the span? Compare the noise level with the weakest signal singular value.

For each utterance and noise level: add iid Gaussian noise of standard deviation sigma to the
first-layer gradient, run the closed-form attack with the length given, and record the error
next to rho = sigma * (sqrt(m) + sqrt(n)) / s_T, where s_T is the smallest singular value that
belongs to the signal and sigma * (sqrt(m) + sqrt(n)) is the spectral norm of an m x n noise matrix.

Usage: python experiments/noise_law.py --feats DIR --out FILE.jsonl [--ckpt PATH]
"""
import argparse, json, sys
import torch
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, client_gradient, text_to_ids
from unsplice.linear_attack import ds1_attack, gradient_rowspace
from unsplice.data import FeatureSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = DeepSpeech1().to(dev)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    fs = FeatureSet(a.feats, "test-clean", "mfcc")
    ids = [i for i in range(len(fs)) if fs.frames(i) <= 300]
    ids = [ids[j] for j in torch.randperm(len(ids), generator=torch.Generator().manual_seed(3)).tolist()][:a.n]
    with open(a.out, "a") as f:
        for i in ids:
            x, text = fs[i]
            x = x.to(dev)
            T = x.shape[0]
            grads, _ = client_gradient(model, x.unsqueeze(1), text_to_ids(text).to(dev))
            W, b = grads["fc1.weight"], grads["fc1.bias"]
            s, _ = gradient_rowspace(W, b)
            m, n = W.shape[0], W.shape[1] + 1
            fro = float(torch.sqrt(W.double().pow(2).sum() + b.double().pow(2).sum()))
            for rho in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0):
                sigma = rho * float(s[T - 1]) / (m ** 0.5 + n ** 0.5)
                noisy = {"fc1.weight": W + torch.randn_like(W) * sigma, "fc1.bias": b + torch.randn_like(b) * sigma}
                X, info = ds1_attack(noisy, T=T)
                err = X - x.double()
                f.write(json.dumps({"id": fs.index[i]["id"], "T": T, "rho": rho, "sigma": sigma,
                                    "sigma_over_rms": sigma / (fro / (m * n) ** 0.5), "s_T_over_s_1": float(s[T - 1] / s[0]),
                                    "rel": float(err.norm() / x.double().norm()), "mae": float(err.abs().mean())}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
