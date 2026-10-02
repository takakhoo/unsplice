"""Attack DeepSpeech-1 on LibriSpeech utterances, one client gradient per utterance.

Closed form for utterances within the first-layer limit, sequential span decoding beyond it.
Writes one JSON line per utterance and, optionally, the reconstructed features.

Usage: python experiments/run_ds1.py --feats DIR --out FILE.jsonl [--ckpt PATH] [--shard i/n]
"""
import argparse, json, os, sys, time, zlib
import torch
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, client_gradient, text_to_ids
from unsplice.linear_attack import ds1_attack
from unsplice.seq_attack import SpanDecoder
from unsplice.data import FeatureSet, boundary_windows, frame_codebook
from unsplice.defenses import apply_defense
from unsplice.metrics import score


def load_model(ckpt, dev, seed=0, n_context=6, relu=True):
    torch.manual_seed(seed)
    model = DeepSpeech1(n_context=n_context, relu=relu).to(dev)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    return model.train()   # cuDNN LSTM backward needs train mode; dropout p stays 0 unless requested


def attack(model, grads, x, method="auto", limit=None, polish=True, inits=None, last_inits=None, known_length=False,
           codebook=None):
    F, c = model.n_feat, model.n_context
    limit = limit or 2 * c * F
    T = x.shape[0]
    t0 = time.time()
    rec = {}
    if method == "linear" or (method == "auto" and T <= limit):
        X, info = ds1_attack(grads, F, c, T=T if known_length else None)
        rec.update(method="linear", T_est=info["T"], rank=info["rank"], residual=info["max_residual"], ambiguity=info["ambiguity"])
    else:
        dec = SpanDecoder(model, grads, codebook=codebook)
        X, info = dec.decode(inits=inits, last_inits=last_inits)
        rec.update(method="sequential", T_est=info["T"], rank=dec.T, **{k: v for k, v in info.items() if k != "T"})
        if polish and len(X):
            rec["chain_mae"] = float((X - x.double()).abs().mean()) if len(X) == T else None
            X, pf = dec.polish(X)
            rec["residual"] = float(pf.max().sqrt())
    rec["seconds"] = time.time() - t0
    rec.update(score(X, x))
    return X, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--min-frames", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=10 ** 9)
    ap.add_argument("--limit", type=int, default=10 ** 9, help="max utterances after filtering")
    ap.add_argument("--method", default="auto")
    ap.add_argument("--defense", default="none")
    ap.add_argument("--save-recon", default=None)
    ap.add_argument("--train-mode", action="store_true", help="client computes the gradient with dropout active")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--known-length", action="store_true", help="give the attacker the utterance length")
    a = ap.parse_args()
    dev = "cuda"
    model = load_model(a.ckpt, dev, a.seed)
    if a.train_mode:
        model.drop.p = a.dropout
    fs = FeatureSet(a.feats, a.split, "mfcc")
    pub = FeatureSet(a.feats, "train-clean-100", "mfcc")     # attacker's public speech, for starting points
    inits = boundary_windows(pub, 48, model.n_context + 1)
    last_inits = boundary_windows(pub, 48, model.n_context + 1, last=True)
    codebook = frame_codebook(pub)
    k, n = map(int, a.shard.split("/"))
    ids = [i for i in range(len(fs)) if a.min_frames <= fs.frames(i) <= a.max_frames]
    g = torch.Generator().manual_seed(1234)
    ids = [ids[j] for j in torch.randperm(len(ids), generator=g).tolist()][:a.limit][k::n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    if a.save_recon:
        os.makedirs(a.save_recon, exist_ok=True)
    with open(a.out, "a") as f:
        for i in ids:
            uid = fs.index[i]["id"]
            if uid in done:
                continue
            x, text = fs[i]
            x = x.to(dev)
            torch.manual_seed(zlib.crc32(uid.encode()))
            grads, loss = client_gradient(model, x.unsqueeze(1), text_to_ids(text).to(dev))
            grads = apply_defense(grads, a.defense)
            X, rec = attack(model, grads, x, a.method, inits=inits, last_inits=last_inits, known_length=a.known_length, codebook=codebook)
            rec.update(id=uid, T=x.shape[0], audio_seconds=fs.index[i]["samples"] / 16000, loss=loss,
                       ckpt=os.path.basename(a.ckpt) if a.ckpt else f"random{a.seed}", defense=a.defense)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if a.save_recon:
                torch.save({"id": uid, "x": x.cpu(), "X": X.float().cpu(), "text": text}, os.path.join(a.save_recon, uid + ".pt"))


if __name__ == "__main__":
    main()
