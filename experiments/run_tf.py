"""Closed-form attack on the Transformer-CTC encoder (pfl4asr front end), one gradient per utterance.

The client normalises its 80-bin log-mel features per utterance, so the attack recovers the
normalised features. Usage: python experiments/run_tf.py --feats DIR --out FILE.jsonl [--ckpt PATH]
"""
import argparse, json, os, sys, time, zlib
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.model import text_to_ids, BLANK
from unsplice.tfctc import TransformerCTC, normalise, KERNEL, STRIDE, PAD
from unsplice.linear_attack import conv1d_attack
from unsplice.metrics import score
from unsplice.data import FeatureSet
from unsplice.defenses import apply_defense


def client_gradient(model, x, y):
    """x: (T, 80) normalised features. Returns gradients of every parameter."""
    logp = model(x.unsqueeze(0)).log_softmax(-1)
    P = logp.shape[1]
    loss = nn.functional.ctc_loss(logp.transpose(0, 1), y.unsqueeze(0), torch.tensor([P]), torch.tensor([len(y)]),
                                  blank=BLANK, zero_infinity=True)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    return {n: g.detach() for (n, _), g in zip(model.named_parameters(), grads)}, float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-frames", type=int, default=10 ** 9)
    ap.add_argument("--min-frames", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--defense", default="none")
    ap.add_argument("--train-mode", action="store_true")
    ap.add_argument("--save-recon", default=None)
    ap.add_argument("--known-length", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = TransformerCTC(n_layers=a.layers).to(dev)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"])
    model.train(a.train_mode)
    fs = FeatureSet(a.feats, a.split, "fbank")
    k, n = map(int, a.shard.split("/"))
    ids = [i for i in range(len(fs)) if a.min_frames <= fs.frames(i) <= a.max_frames]
    g = torch.Generator().manual_seed(1234)
    ids = [ids[j] for j in torch.randperm(len(ids), generator=g).tolist()][:a.limit][k::n]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    if a.save_recon:
        os.makedirs(a.save_recon, exist_ok=True)
    with open(a.out, "a") as f:
        for i in ids:
            uid = fs.index[i]["id"]
            if uid in done:
                continue
            x, text = fs[i]
            x = normalise(x.unsqueeze(0)).squeeze(0).to(dev)
            T = x.shape[0]
            torch.manual_seed(zlib.crc32(uid.encode()))
            grads, loss = client_gradient(model, x, text_to_ids(text).to(dev))
            grads = apply_defense(grads, a.defense)
            t0 = time.time()
            X, info = conv1d_attack(grads["conv.weight"], grads["conv.bias"], 80, KERNEL, STRIDE, PAD,
                                    T=T if a.known_length else None)
            rec = {"id": uid, "T": T, "T_est": info["T"], "rank": info["rank"], "seconds": time.time() - t0,
                   "loss": loss, "residual": info["max_residual"], "ambiguity": info["ambiguity"], "audio_seconds": fs.index[i]["samples"] / 16000,
                   "ckpt": os.path.basename(a.ckpt) if a.ckpt else "random0", "defense": a.defense}
            rec.update(score(X, x))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if a.save_recon:
                torch.save({"id": uid, "x": x.cpu(), "X": X.float().cpu(), "text": text}, os.path.join(a.save_recon, uid + ".pt"))


if __name__ == "__main__":
    main()
