"""Train the feature-to-mel network that lets a pretrained HiFi-GAN speak recovered features.

Usage: python experiments/train_feat2mel.py --src mfcc|fbank --feats DIR --libri ROOT --out DIR
"""
import argparse, json, os, random, sys, time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.audio import Feat2Mel, hifigan_mel
from unsplice.data import FeatureSet, list_librispeech
from unsplice.tfctc import normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--libri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="train-clean-100")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    sets = []
    for split in a.train.split(","):
        fs = FeatureSet(a.feats, split, a.src)
        paths = {u: p for u, p, _ in list_librispeech(a.libri, split)}
        sets.append((fs, paths))
    items = [(k, i) for k, (fs, _) in enumerate(sets) for i in range(len(fs)) if fs.index[i]["samples"] <= 16000 * 17]
    model = Feat2Mel(26 if a.src == "mfcc" else 80).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0, run = time.time(), 0.0
    for step in range(1, a.steps + 1):
        batch = random.sample(items, a.batch)
        feats, wavs = [], []
        for k, i in batch:
            fs, paths = sets[k]
            f, _ = fs[i]
            w, _ = sf.read(paths[fs.index[i]["id"]], dtype="float32")
            feats.append(f)
            wavs.append(torch.from_numpy(w))
        flen = torch.tensor([len(f) for f in feats])
        wlen = torch.tensor([len(w) for w in wavs])
        x = nn.utils.rnn.pad_sequence(feats, batch_first=True).to(dev)
        if a.src == "fbank":
            x = normalise(x, flen)
        wav = nn.utils.rnn.pad_sequence(wavs, batch_first=True).to(dev)
        with torch.no_grad():
            mel = hifigan_mel(wav)
        mlen = (wlen // 256 + 1).to(dev)
        pred = model(x, mel.shape[2], mlen)
        mask = (torch.arange(mel.shape[2], device=dev).unsqueeze(0) < mlen.unsqueeze(1)).unsqueeze(1)
        loss = ((pred - mel).abs() * mask).sum() / (mask.sum() * mel.shape[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += float(loss.detach())
        if step % 500 == 0:
            rec = {"step": step, "l1": run / 500, "minutes": (time.time() - t0) / 60}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = 0.0
        if step % 5000 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "src": a.src, "step": step}, os.path.join(a.out, "feat2mel.pt"))


if __name__ == "__main__":
    main()
