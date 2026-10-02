"""Prior-art baseline: recover features by gradient matching (optimise X until its gradient matches).

Follows the earlier DeepSpeech attack: known transcript and length, cosine distance, Adam with
lr 0.5 halved every 250 steps, uniform initialisation. `--layers last` matches only the output
layer (as that code did); `--layers all` matches every parameter tensor.

Usage: python experiments/baseline_gm.py --feats DIR --out FILE.jsonl [--relu 0|1] [--layers last|all]
"""
import argparse, json, sys, time
import torch
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, text_to_ids
from unsplice.ctc2 import ctc_loss_imp
from unsplice.data import FeatureSet
from unsplice.metrics import feature_metrics


def grads_of(model, x, y, params, create_graph):
    lp = model(x.unsqueeze(1)).log_softmax(-1)
    loss = ctc_loss_imp(lp, y.unsqueeze(0), [lp.shape[0]], [len(y)])
    return torch.autograd.grad(loss, params, create_graph=create_graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--relu", type=int, default=1)
    ap.add_argument("--layers", default="last")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = DeepSpeech1(relu=bool(a.relu)).to(dev)
    params = [model.out.weight, model.out.bias] if a.layers == "last" else list(model.parameters())
    fs = FeatureSet(a.feats, "test-clean", "mfcc")
    ids = [i for i in range(len(fs)) if fs.frames(i) <= a.max_frames]
    g = torch.Generator().manual_seed(1234)
    k, n = map(int, a.shard.split("/"))
    ids = [ids[j] for j in torch.randperm(len(ids), generator=g).tolist()][:a.n][k::n]
    with torch.backends.cudnn.flags(enabled=False), open(a.out, "a") as f:
        for i in ids:
            x, text = fs[i]
            x, y = x.to(dev), text_to_ids(text).to(dev)
            target = [t.detach() for t in grads_of(model, x, y, params, False)]
            torch.manual_seed(i)
            X = torch.nn.Parameter((torch.rand_like(x) * 2 - 1))
            opt = torch.optim.Adam([X], lr=a.lr)
            sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(range(250, a.steps, 250)), gamma=0.5)
            t0, curve = time.time(), []
            for step in range(a.steps):
                gs = grads_of(model, X, y, params, True)
                dist = sum(1 - torch.nn.functional.cosine_similarity(u.reshape(1, -1), v.reshape(1, -1)) for u, v in zip(gs, target)).squeeze()
                opt.zero_grad()
                dist.backward()
                opt.step()
                sched.step()
                if step % 100 == 0 or step == a.steps - 1:
                    curve.append({"step": step, "dist": float(dist.detach()), "mae": float((X.detach() - x).abs().mean()),
                                  "seconds": time.time() - t0})
            rec = {"id": fs.index[i]["id"], "T": x.shape[0], "relu": a.relu, "layers": a.layers, "steps": a.steps,
                   "seconds": time.time() - t0, "final_dist": curve[-1]["dist"], "curve": curve,
                   **feature_metrics(X.detach(), x)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print({k: v for k, v in rec.items() if k != "curve"}, flush=True)


if __name__ == "__main__":
    main()
