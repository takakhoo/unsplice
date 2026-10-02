"""Collect what the demo page needs for a few utterances: audio files, transcripts, and feature arrays.

Usage: python experiments/export_demo.py <audio eval dir> <recon dir> <out dir> [n]
"""
import glob, json, os, shutil, sys
import torch

aud, recon, out = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
os.makedirs(out, exist_ok=True)
rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(aud, "per_utterance.jsonl"))}
ids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(aud, "wav", "*_reconstructed.wav"))})
ids = [u for u in ids if u in rows][:n]
items = []
for uid in ids:
    d = torch.load(os.path.join(recon, uid + ".pt"))
    for kind in ("original", "reconstructed", "griffinlim"):
        src = os.path.join(aud, "wav", f"{uid}_{kind}.wav")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f"{uid}_{kind}.wav"))
    r = rows[uid]
    items.append({"id": uid, "seconds": r["seconds"], "ref": r["ref"], "hyp_rec": r["hyp_rec"], "hyp_orig": r["hyp_orig"],
                  "hyp_gl": r.get("hyp_gl"), "spk_sim": r["spk_sim_rec"], "feat_mae": r["feat_mae"],
                  "x": [[round(float(v), 2) for v in row] for row in d["x"][:, :26]],
                  "X": [[round(float(v), 2) for v in row] for row in d["X"][:, :26]]})
json.dump({"items": items}, open(os.path.join(out, "demo.json"), "w"))
print([i["id"] for i in items])
