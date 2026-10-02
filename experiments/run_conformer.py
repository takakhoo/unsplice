"""Span decoding against a Conformer / ESPnet style front end (Conv2dSubsampling + Linear).

The first convolutions have too few distinct patch directions to leak anything by themselves,
but the linear projection after subsampling sees one d_in-dimensional vector per output frame,
and its gradient leaks their span as long as the number of output frames stays below d_model.

Usage: python experiments/run_conformer.py --feats DIR --out FILE.jsonl [--d 256] [--limit N]
"""
import argparse, json, sys, time
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.local_decode import LocalSpanDecoder
from unsplice.linear_attack import gradient_rowspace, estimate_rank
from unsplice.data import FeatureSet
from unsplice.tfctc import normalise
from unsplice.model import text_to_ids, BLANK
from unsplice.metrics import feature_metrics

K, S = 7, 4          # one output frame reads 7 input frames; consecutive output frames are 4 apart


class ConformerFrontCTC(nn.Module):
    """Conv2dSubsampling (two 3x3 stride-2 convs, ReLU), Linear to d_model, then a Transformer encoder and CTC head."""

    def __init__(self, d=256, n_mels=80, n_layers=4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d, 3, 2), nn.ReLU(), nn.Conv2d(d, d, 3, 2), nn.ReLU())
        self.fdim = ((n_mels - 1) // 2 - 1) // 2
        self.proj = nn.Linear(d * self.fdim, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 29)

    def lift(self, x):
        """x: (T, F) -> (T', d * F') inputs of the projection."""
        z = self.conv(x.unsqueeze(0).unsqueeze(0))                       # (1, d, T', F')
        return z.permute(0, 2, 1, 3).reshape(z.shape[2], -1)

    def forward(self, x):
        h = self.proj(self.lift(x))
        return self.out(self.enc(h.unsqueeze(0))).squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--min-frames", type=int, default=150)
    ap.add_argument("--inits", type=int, default=10)
    ap.add_argument("--single", action="store_true", help="one seed and one chain instead of several segments")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = ConformerFrontCTC(a.d).to(dev)
    dmodel = ConformerFrontCTC(a.d).to(dev).double()
    dmodel.load_state_dict(model.state_dict())
    for p in dmodel.parameters():      # the attacker's copy is only evaluated, never trained
        p.requires_grad_(False)
    fs = FeatureSet(a.feats, "test-clean", "fbank")
    pub = FeatureSet(a.feats, "train-clean-100", "fbank")
    g = torch.Generator().manual_seed(5)
    inits = []
    for i in torch.randperm(len(pub), generator=g).tolist()[:40]:
        x = normalise(pub[i][0].unsqueeze(0)).squeeze(0)
        t = int(torch.randint(0, len(x) - K, (1,), generator=g))
        inits.append(x[t:t + K])
    inits = torch.stack(inits).to(dev)
    ids = [i for i in range(len(fs)) if a.min_frames <= fs.frames(i) <= a.max_frames]
    ids = [ids[j] for j in torch.randperm(len(ids), generator=torch.Generator().manual_seed(1234)).tolist()][:a.limit]
    with open(a.out, "a") as f:
        for i in ids:
            x, text = fs[i]
            x = normalise(x.unsqueeze(0)).squeeze(0).to(dev)
            T = x.shape[0]
            logp = model(x).log_softmax(-1)
            y = text_to_ids(text).to(dev)
            loss = nn.functional.ctc_loss(logp.unsqueeze(1), y.unsqueeze(0), torch.tensor([logp.shape[0]]), torch.tensor([len(y)]),
                                          blank=BLANK, zero_infinity=True)
            gW, gb = torch.autograd.grad(loss, [model.proj.weight, model.proj.bias])
            t0 = time.time()
            s, vh = gradient_rowspace(gW, gb)
            rank = estimate_rank(s)
            dec = LocalSpanDecoder(lambda w: dmodel.lift(w).squeeze(0), vh[:rank], K, S, 80, phi_all=dmodel.lift)
            if a.single:
                X1, info = dec.decode(inits[:a.inits])
                segments = [] if X1 is None else [X1]
            else:
                segments, info = dec.decode_segments(inits[:a.inits], max_segments=6)
            rec = {"id": fs.index[i]["id"], "T": T, "frames_out": logp.shape[0], "rank": rank, "d": a.d,
                   "seconds": time.time() - t0, "n_segments": len(segments), **info}
            # Evaluation only: place each segment where it matches the true features best.
            usable = (logp.shape[0] - 1) * S + K               # frames the front end actually reads
            covered = torch.zeros(T, dtype=torch.bool)
            err_sum, err_n, sig = 0.0, 0, 0.0
            for seg in segments:
                n = seg.shape[0]
                if n > T:
                    continue
                win = x.double().unfold(0, n, 1).permute(0, 2, 1)
                d = (win - seg).abs().mean(dim=(1, 2))
                o = int(d.argmin())
                if float(d[o]) < 0.1:                          # a real match, not a spurious segment
                    new = ~covered[o:o + n]
                    e = (seg - x[o:o + n].double())[new.to(dev)]
                    err_sum += float(e.abs().sum()); err_n += e.numel()
                    sig += float(x[o:o + n].double()[new.to(dev)].pow(2).sum())
                    rec.setdefault("sq", 0.0); rec["sq"] += float(e.pow(2).sum())
                    covered[o:o + n] = True
            rec["coverage"] = float(covered[:usable].float().mean())
            if err_n:
                rec["mae"] = err_sum / err_n
                rec["snr_db"] = float(10 * torch.log10(torch.tensor(sig / max(rec.pop("sq"), 1e-300))))
            rec["segment_lengths"] = [int(seg.shape[0]) for seg in segments]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)


if __name__ == "__main__":
    main()
