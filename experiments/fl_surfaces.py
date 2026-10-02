"""What a federated server actually receives: batches, local SGD steps, and sums over clients.

Each scenario builds the update the server would see for DeepSpeech-1 and runs the closed-form
first-layer attack on it. The first-layer update is always sum_k sum_t g_t^(k) c_t^T, whatever the
local weights were at step k, so its row space is the span of the context windows of every
utterance that touched it.

Usage: python experiments/fl_surfaces.py --feats DIR --out FILE.jsonl [--ckpt PATH]
"""
import argparse, copy, itertools, json, sys, time
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, text_to_ids, BLANK
from unsplice.linear_attack import linear_span_attack, gradient_rowspace, estimate_rank
from unsplice.data import FeatureSet


def utt_loss(model, x, y):
    z = model(x.unsqueeze(1))
    return nn.functional.ctc_loss(z.log_softmax(-1), y.unsqueeze(0), torch.tensor([z.shape[0]]),
                                  torch.tensor([len(y)]), blank=BLANK)


def local_update(model, utts, steps, lr, batch, optim="sgd", momentum=0.0, weight_decay=0.0):
    """Run local training and return what the server can compute from the returned weights.

    For SGD the server undoes weight decay exactly (it knows lr, the decay and the starting
    weights), which leaves a weighted sum of the local gradients.
    """
    work = copy.deepcopy(model)
    if optim == "adam":
        opt = torch.optim.Adam(work.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(work.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    order = list(range(len(utts)))
    k = 0
    for _ in range(steps):
        chosen = order if batch >= len(utts) else [order[(k + j) % len(utts)] for j in range(batch)]
        k += batch
        opt.zero_grad()
        loss = sum(utt_loss(work, *utts[i]) for i in chosen) / len(chosen)
        loss.backward()
        opt.step()
    before = dict(model.named_parameters())
    shrink = 1.0
    if optim == "sgd" and weight_decay and not momentum:
        shrink = (1 - lr * weight_decay) ** steps
    return {n: ((shrink * before[n] - p) / lr).detach() for n, p in work.named_parameters()}


def attack(update, lengths, F=26, c=6):
    t0 = time.time()
    s, vh = gradient_rowspace(update["fc1.weight"], update["fc1.bias"])
    rank = estimate_rank(s)
    # Lengths are given to the attacker here, so the span dimension is their sum (capped by the layer).
    total = min(sum(lengths), vh.shape[0])
    X, info = linear_span_attack(update["fc1.weight"], update["fc1.bias"], F, 2 * c + 1, 1, c,
                                 T=[tuple(lengths)], rank=total)
    return X, {"rank": rank, "residual": info["max_residual"], "seconds": time.time() - t0}


def pick(fs, n, total_max, gen, crop=None):
    """n utterances whose frame counts add up to at most total_max.

    With crop set, each utterance is cut to its first `crop` frames (a short voice command)
    and its transcript shortened in proportion.
    """
    cap = 10 ** 9 if crop else total_max // n
    ids = [i for i in range(len(fs)) if fs.frames(i) <= cap]
    perm = torch.randperm(len(ids), generator=gen).tolist()
    return [ids[j] for j in perm[:n]]


def load(fs, i, dev, crop=None):
    x, text = fs[i]
    if crop and len(x) > crop:
        text = text[:max(2, int(len(text) * crop / len(x)))]
        x = x[:crop]
    return x.to(dev), text_to_ids(text).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = DeepSpeech1().to(dev)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.train()
    fs = FeatureSet(a.feats, "test-clean", "mfcc")
    gen = torch.Generator().manual_seed(7)
    scenarios = []
    for steps, lr in [(1, 0.01), (2, 0.01), (5, 0.01), (10, 0.01), (10, 0.1), (50, 0.01)]:
        scenarios.append({"name": "local_sgd", "n_utts": 1, "steps": steps, "lr": lr, "batch": 1, "budget": 300})
    scenarios.append({"name": "sgd_momentum", "n_utts": 1, "steps": 10, "lr": 0.1, "batch": 1, "budget": 300, "momentum": 0.9})
    scenarios.append({"name": "sgd_weight_decay", "n_utts": 1, "steps": 10, "lr": 0.1, "batch": 1, "budget": 300, "weight_decay": 1e-3})
    scenarios.append({"name": "adam", "n_utts": 1, "steps": 1, "lr": 1e-3, "batch": 1, "budget": 300, "optim": "adam"})
    scenarios.append({"name": "adam", "n_utts": 1, "steps": 10, "lr": 1e-3, "batch": 1, "budget": 300, "optim": "adam"})
    scenarios.append({"name": "batch", "n_utts": 2, "steps": 1, "lr": 0.1, "batch": 2, "budget": 300})
    for B, crop in [(4, 60), (8, 35), (16, 18)]:
        scenarios.append({"name": "batch_short", "n_utts": B, "steps": 1, "lr": 0.1, "batch": B, "budget": 300, "crop": crop})
    for B, crop, steps in [(4, 60, 8), (8, 35, 16)]:
        scenarios.append({"name": "local_epochs_batch1", "n_utts": B, "steps": steps, "lr": 0.1, "batch": 1,
                          "budget": 300, "crop": crop})
    for B, crop in [(4, 90), (8, 60)]:
        scenarios.append({"name": "over_budget", "n_utts": B, "steps": 1, "lr": 0.1, "batch": B, "budget": 600, "crop": crop})
    with open(a.out, "a") as f:
        for sc in scenarios:
            if a.only and sc["name"] not in a.only.split(","):
                continue
            for trial in range(a.trials):
                ids = pick(fs, sc["n_utts"], sc["budget"], gen, sc.get("crop"))
                # Distinct lengths: with equal lengths any reordering of the utterances also solves the
                # system, and a linear solver returns a blend of the reorderings.
                crops = [sc["crop"] - len(ids) + 2 * j for j in range(len(ids))] if sc.get("crop") else [None] * len(ids)
                utts = [load(fs, i, dev, cr) for i, cr in zip(ids, crops)]
                total = sum(u[0].shape[0] for u in utts)
                update = local_update(model, utts, sc["steps"], sc["lr"], sc["batch"], sc.get("optim", "sgd"),
                                      sc.get("momentum", 0.0), sc.get("weight_decay", 0.0))
                lengths = [u[0].shape[0] for u in utts]
                X, rec = attack(update, lengths)
                x = torch.cat([u[0] for u in utts]).double()
                err = X - x
                rec.update(sc, trial=trial, total_frames=total, lengths=lengths, mae=float(err.abs().mean()),
                           rel=float(err.norm() / x.norm()), rank_ok=rec["rank"] == total,
                           ckpt=a.ckpt or "random0")
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(sc, "done", flush=True)


if __name__ == "__main__":
    main()
