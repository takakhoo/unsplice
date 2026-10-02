"""Build the interactive page (docs/index.html) from results and demo audio. Usage: python experiments/build_page.py"""
import base64, collections, glob, json, os, statistics as st

S = json.load(open("results/summary.json"))
R = "results/raw"


def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(R, pat))) for l in open(f) if l.strip()]


def audio_uri(path):
    return "data:audio/wav;base64," + base64.b64encode(open(path, "rb").read()).decode() if os.path.exists(path) else None


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


a, b, c = S["ds1_closed_form"], S["ds1_sequential"], S["tf_closed_form"]
aud = S["audio"].get("ds1_short") or S["audio"].get("mfcc_prelim")
D = {"stats": [
    {"value": pct(a["success_rate"], 1), "label": f"of {a['n']:,} utterances up to 6.2 s recovered in closed form (DeepSpeech-1)"},
    {"value": f"{b['seconds_of_audio_max']:.0f} s", "label": f"longest utterance recovered; {pct(b['success_rate'])} of {b['n']} long utterances"},
    {"value": pct(aud["wer"]["rec"], 1), "label": f"Whisper word error rate on the reconstructed audio (originals: {pct(aud['wer']['orig'], 1)})"},
    {"value": pct(aud["speaker_id_top1"]), "label": f"of reconstructions assigned to the right speaker out of {aud.get('enrolled_speakers', aud.get('speakers'))}"},
]}
demo = []
for folder, method in (("docs/audio/short", "closed form"), ("docs/audio/long", "sequential")):
    p = os.path.join(folder, "demo.json")
    if not os.path.exists(p):
        continue
    for it in json.load(open(p))["items"]:
        it["method"] = method
        it["audio"] = {k: audio_uri(os.path.join(folder, f"{it['id']}_{k}.wav")) for k in ("original", "reconstructed", "griffinlim")}
        demo.append(it)
D["demo"] = demo
e1, e2 = load("e1/random0_linear.*.jsonl"), (load("e2b/*.jsonl") or load("e2/*.jsonl"))
seen = set()
e2 = [r for r in e2 if not (r["id"] in seen or seen.add(r["id"]))]
row = lambda r: [round(r["audio_seconds"], 2), float(f"{max(r['mae'], 1e-9):.2e}"), round(r["seconds"], 1)]
D["scatter"] = {"closed": [row(r) for r in e1 if r["mae"] < 1e-2], "sequential": [row(r) for r in e2 if r["mae"] < 1e-2],
                "ambiguous": [row(r) for r in e1 + e2 if r["mae"] >= 1e-2],
                "baseline": [[round(g["T"] * 0.02, 2), float(f"{g['mae']:.2e}"), round(g["seconds"], 0)] for g in S["baseline_gradient_matching"]]}
D["cliff"] = []
for name, v in S["capacity_cliff"].items():
    cap = v["predicted"]
    pts = [[int(P) - cap, float(f"{e:.2e}")] for P, e in v["median_rel_by_patches"].items() if abs(int(P) - cap) <= 24]
    D["cliff"].append({"name": name.split(" (")[0] + f" ({cap})", "points": sorted(pts)})
D["noise"] = [[float(k), float(f"{v['median_rel']:.2e}")] for k, v in sorted(S["noise_law"].items(), key=lambda kv: float(kv[0]))]
order = [("local_sgd|utts=1|steps=1|lr=0.01", "1 local SGD step"), ("local_sgd|utts=1|steps=10|lr=0.01", "10 local SGD steps"),
         ("local_sgd|utts=1|steps=50|lr=0.01", "50 local SGD steps"), ("local_sgd|utts=1|steps=10|lr=0.1", "10 SGD steps, learning rate 0.1"),
         ("sgd_momentum|utts=1|steps=10|lr=0.1", "SGD with momentum 0.9"), ("sgd_weight_decay|utts=1|steps=10|lr=0.1", "SGD with weight decay"),
         ("batch|utts=2|steps=1|lr=0.1", "Batch of 2 utterances"), ("batch_short|utts=4|steps=1|lr=0.1", "Batch of 4 short utterances"),
         ("batch_short|utts=8|steps=1|lr=0.1", "Batch of 8 short utterances"), ("batch_short|utts=16|steps=1|lr=0.1", "Batch of 16 short utterances"),
         ("local_epochs_batch1|utts=8|steps=16|lr=0.1", "8 utterances, 2 local epochs, batch 1"),
         ("over_budget|utts=4|steps=1|lr=0.1", "Batch over capacity (4 utterances)"), ("over_budget|utts=8|steps=1|lr=0.1", "Batch over capacity (8 utterances)"),
         ("adam|utts=1|steps=1|lr=0.001", "1 local Adam step"), ("adam|utts=1|steps=10|lr=0.001", "10 local Adam steps")]
D["fl"] = [[lab, int(S["fl"][k]["median_total_frames"]), S["fl"][k]["median_mae"], S["fl"][k]["success_rate"]] for k, lab in order if k in S["fl"]]
for k, lab in (("ds1_dropout0.2_linear", "Dropout 0.2 on the client"), ("ds1_dropout0.5_linear", "Dropout 0.5 on the client"),
               ("ds1_clip_0.01_blind", "Update clipped to norm 0.01"), ("ds1_f16_blind", "Update sent in half precision"),
               ("ds1_quant_8_known", "8-bit quantised update"), ("ds1_prune_0.5_known", "Half the entries pruned"), ("ds1_sign_known", "Sign of the update only")):
    v = S["defenses"].get(k)
    if v:
        D["fl"].append([lab, "up to 312", v["median_mae"], v["success_rate"]])
D["probe"] = S.get("architecture_probe", {}).get("rows", [])
D["limits"] = [
    f"Digital silence. Runs of 13 or more bit-identical frames make windows identical and the solution non-unique. The solver flags these; they are {pct(a['failures'] / a['n'], 1)} of the short utterances and count as failures.",
    "Capacity. The closed form stops at (k − s) × F windows in total across everything in one update. Deeper layers extend it to the width of those layers.",
    "Features only. The attack recovers what the front end computes. 26 MFCCs carry no pitch, so the vocoder infers it.",
    "Local Adam, pruning and sign compression break the low-rank structure, and enough noise hides the span.",
    "Whisper, wav2vec 2.0, DeepSpeech-2 and QuartzNet first layers do not leak a usable span. No attack on them is claimed.",
    "Secure aggregation over enough clients exceeds every capacity measured here.",
]
tpl = open("docs/page.template.html").read().replace("/*DATA*/null", json.dumps(D, separators=(",", ":")))
os.makedirs("docs", exist_ok=True)
open("docs/unsplice_page.html", "w").write(tpl)                      # body-only version for publishing as an artifact
open("docs/index.html", "w").write('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                                   + tpl.split("<div class=\"wrap\">")[0] + "</head>\n<body>\n<div class=\"wrap\">" + tpl.split("<div class=\"wrap\">", 1)[1] + "\n</body>\n</html>\n")
print("demo items:", len(demo), "| page size: %.1f MB" % (len(tpl) / 1e6))
