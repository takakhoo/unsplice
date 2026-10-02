"""Train a CTC acoustic model on LibriSpeech so the attacks can be run against trained weights.

Usage: python experiments/train_asr.py --model ds1|tf --feats DIR --out DIR [--epochs N]
Saves checkpoints at step 0, a few early steps, and selected epochs, with dev-clean WER in log.jsonl.
"""
import argparse, json, math, os, random, sys, time
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, text_to_ids, ids_to_text, BLANK
from unsplice.tfctc import TransformerCTC, normalise
from unsplice.data import FeatureSet


def batches(fs, budget, shuffle, max_frames):
    order = sorted((i for i in range(len(fs)) if fs.frames(i) <= max_frames), key=fs.frames)
    out, cur, longest = [], [], 0
    for i in order:
        longest_new = max(longest, fs.frames(i))
        if cur and longest_new * (len(cur) + 1) > budget:
            out.append(cur)
            cur, longest_new = [], fs.frames(i)
        cur.append(i)
        longest = longest_new
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(fs, idxs, dev):
    feats, texts = zip(*(fs[i] for i in idxs))
    lens = torch.tensor([len(f) for f in feats])
    x = nn.utils.rnn.pad_sequence(feats, batch_first=True).to(dev)
    ys = [text_to_ids(t) for t in texts]
    return x, lens, ys, texts


def greedy(logp, lens):
    """logp: (B, T, C). Collapse repeats and drop blanks."""
    out = []
    for b, n in enumerate(lens):
        ids = logp[b, :n].argmax(-1).tolist()
        prev, seq = BLANK, []
        for k in ids:
            if k != prev and k != BLANK:
                seq.append(k)
            prev = k
        out.append(ids_to_text(seq))
    return out


def spec_augment(x, lens, n_freq=2, f_width=20, n_time=4, t_frac=0.05):
    x = x.clone()
    for b, n in enumerate(lens):
        for _ in range(n_freq):
            w = random.randint(0, f_width)
            f0 = random.randint(0, x.shape[2] - w)
            x[b, :, f0:f0 + w] = 0
        for _ in range(n_time):
            w = random.randint(0, max(1, int(t_frac * n)))
            t0 = random.randint(0, max(0, int(n) - w))
            x[b, t0:t0 + w] = 0
    return x


def run(model, kind, x, lens):
    """Returns (log-probs (B, T', C), output lengths)."""
    if kind == "ds1":
        return model(x.transpose(0, 1)).transpose(0, 1).log_softmax(-1), lens
    return model(x, lens.to(x.device)).log_softmax(-1), TransformerCTC.out_lengths(lens)


@torch.no_grad()
def evaluate(model, kind, fs, dev, n=600, budget=40000):
    import jiwer
    model.eval()
    idx = list(range(0, len(fs), max(1, len(fs) // n)))[:n]
    refs, hyps = [], []
    for k in range(0, len(idx), 16):
        x, lens, ys, texts = collate(fs, idx[k:k + 16], dev)
        if kind == "tf":
            x = normalise(x, lens)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=kind == "tf"):
            logp, olens = run(model, kind, x, lens)
        hyps += greedy(logp.float(), olens)
        refs += list(texts)
    model.train()
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", default="train-clean-100")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    kind = a.model
    feat_kind = "mfcc" if kind == "ds1" else "fbank"
    train_sets = [FeatureSet(a.feats, s, feat_kind) for s in a.train.split(",")]
    dev_set = FeatureSet(a.feats, "dev-clean", feat_kind)
    if kind == "ds1":
        model = DeepSpeech1(dropout=a.dropout).to(dev)
        budget, max_frames, lr, warm = 16000, 900, a.lr or 2e-4, 500
    else:
        model = TransformerCTC(n_layers=a.layers, dropout=a.dropout).to(dev)
        budget, max_frames, lr, warm = 60000, 1800, a.lr or 4e-4, 4000
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2 if kind == "tf" else 0.0, betas=(0.9, 0.98))
    n_batches = sum(len(batches(fs, budget, False, max_frames)) for fs in train_sets)
    total = n_batches * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    def save(tag, step, extra):
        torch.save({"model": model.state_dict(), "step": step, "kind": kind, "args": vars(a), **extra},
                   os.path.join(a.out, f"ckpt_{tag}.pt"))

    wer, cer = evaluate(model, kind, dev_set, dev)
    save("step0", 0, {"wer": wer})
    log.write(json.dumps({"step": 0, "epoch": 0, "wer": wer, "cer": cer}) + "\n")
    log.flush()
    step, t0 = 0, time.time()
    early = {100, 300, 1000, 3000}
    keep_epochs = {1, 2, 3, 5, 8, 12, 20, 30, 40, 60, a.epochs}
    for ep in range(1, a.epochs + 1):
        all_b = [(fs, b) for fs in train_sets for b in batches(fs, budget, True, max_frames)]
        random.shuffle(all_b)
        run_loss, n_seen = 0.0, 0
        for fs, b in all_b:
            x, lens, ys, _ = collate(fs, b, dev)
            if kind == "tf":
                x = spec_augment(normalise(x, lens), lens)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=kind == "tf"):
                logp, olens = run(model, kind, x, lens)
            loss = nn.functional.ctc_loss(logp.float().transpose(0, 1), nn.utils.rnn.pad_sequence(ys, batch_first=True),
                                          olens, torch.tensor([len(y) for y in ys]), blank=BLANK, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            run_loss += float(loss)
            n_seen += 1
            if step in early:
                wer, cer = evaluate(model, kind, dev_set, dev)
                save(f"step{step}", step, {"wer": wer})
                log.write(json.dumps({"step": step, "epoch": ep, "loss": run_loss / n_seen, "wer": wer, "cer": cer}) + "\n")
                log.flush()
        wer, cer = evaluate(model, kind, dev_set, dev)
        rec = {"step": step, "epoch": ep, "loss": run_loss / n_seen, "wer": wer, "cer": cer, "minutes": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        save("last", step, {"wer": wer, "epoch": ep})
        if ep in keep_epochs:
            save(f"ep{ep}", step, {"wer": wer, "epoch": ep})


if __name__ == "__main__":
    main()
