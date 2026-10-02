"""Small arrays the figures need: singular value spectra and a few reconstructions."""
import glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, ".")
from unsplice.model import DeepSpeech1, client_gradient, text_to_ids
from unsplice.linear_attack import gradient_rowspace
from unsplice.deep_attack import leaked_spans
from unsplice.data import FeatureSet

out = os.path.join(os.environ["OUT"], "figdata")
os.makedirs(out, exist_ok=True)
dev = "cuda"
fs = FeatureSet(os.environ["FEATS"], "test-clean", "mfcc")
torch.manual_seed(0)
model = DeepSpeech1().to(dev)
spec = {}
for target in (150, 250, 500, 1000):
    i = min(range(len(fs)), key=lambda i: abs(fs.frames(i) - target))
    x, text = fs[i]
    g, _ = client_gradient(model, x.to(dev).unsqueeze(1), text_to_ids(text).to(dev))
    s, _ = gradient_rowspace(g["fc1.weight"], g["fc1.bias"])
    entry = {"T": int(x.shape[0]), "fc1": (s / s[0]).tolist()}
    if target >= 500:
        _, _, sv = leaked_spans(g)
        for k, v in sv.items():
            entry[k] = (v / v[0]).tolist()[:2100]
    spec[fs.index[i]["id"]] = entry
json.dump(spec, open(os.path.join(out, "spectra.json"), "w"))
ex = {}
short = sorted(glob.glob(os.path.join(os.environ["OUT"], "e1/recon/*.pt")))
long_ = sorted(glob.glob(os.path.join(os.environ["OUT"], "e2/recon/*.pt")), key=lambda p: -torch.load(p)["x"].shape[0])
for tag, p in (("short", short[7]), ("long", long_[0])):
    d = torch.load(p)
    ex[tag + "_x"], ex[tag + "_X"] = d["x"].numpy(), d["X"].numpy()
    ex[tag + "_id"], ex[tag + "_text"] = d["id"], d["text"]
np.savez_compressed(os.path.join(out, "examples.npz"), **ex)
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in ex.items()})
