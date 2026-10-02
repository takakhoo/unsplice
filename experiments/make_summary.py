"""Collect every result table into results/summary.json, the single source for numbers quoted in the README and paper.

Usage: python experiments/make_summary.py [results/raw]
"""
import collections, glob, json, os, statistics as st, sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/raw"


def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(R, pat))) for l in open(f) if l.strip()]


def q(vals, p):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(p * len(vals)))] if vals else None


def block(rows, ok=lambda r: r["mae"] < 1e-2, snr_key="snr_db"):
    if not rows:
        return None
    return {"n": len(rows), "success_rate": sum(ok(r) for r in rows) / len(rows),
            "median_mae": st.median(r["mae"] for r in rows), "p90_mae": q([r["mae"] for r in rows], 0.9),
            "median_snr_db": st.median(r[snr_key] for r in rows), "median_seconds": st.median(r["seconds"] for r in rows),
            "length_exact_rate": sum(r.get("length_error", 0) == 0 for r in rows) / len(rows)}


S = {}
e1 = load("e1/random0_linear.*.jsonl")
S["ds1_closed_form"] = block(e1)
if e1:
    fails = [r for r in e1 if r["mae"] >= 1e-2]
    S["ds1_closed_form"].update(
        failures=len(fails), failures_flagged_ambiguous=sum((r.get("ambiguity") or 0) > 1e-2 for r in fails),
        failure_speakers=dict(collections.Counter(r["id"].split("-")[0] for r in fails)),
        seconds_of_audio_min=min(r["audio_seconds"] for r in e1), seconds_of_audio_max=max(r["audio_seconds"] for r in e1))
e2 = load("e2b/*.jsonl") or load("e2/*.jsonl")     # e2b: decoder with the fc2 and fc3 spans (the default); e2: first run, with the LSTM span as well
seen, e2u = set(), []
for r in e2:
    if r["id"] not in seen:
        seen.add(r["id"])
        e2u.append(r)
S["ds1_sequential"] = block(e2u)
old = load("e2/*.jsonl")
if old and load("e2b/*.jsonl"):
    S["ds1_sequential_with_lstm_span"] = block(old)
if e2u:
    fails = [r for r in e2u if r["mae"] >= 1e-2]
    S["ds1_sequential"].update(failures=len(fails), failure_ids=[r["id"] for r in fails],
                               seconds_of_audio_min=min(r["audio_seconds"] for r in e2u),
                               seconds_of_audio_max=max(r["audio_seconds"] for r in e2u),
                               longest={k: max(e2u, key=lambda r: r["T"])[k] for k in ("id", "T", "audio_seconds", "mae", "snr_db", "seconds")},
                               forward_first_try=sum(r.get("first_tries") == 1 for r in e2u) / len(e2u))
tf = load("tfatk/random0.*.jsonl")
S["tf_closed_form"] = block(tf, ok=lambda r: r["snr_db"] > 30)
if tf:
    fails = [r for r in tf if r["snr_db"] <= 30]
    S["tf_closed_form"].update(failures=len(fails), failure_speakers=dict(collections.Counter(r["id"].split("-")[0] for r in fails)),
                               seconds_of_audio_max=max(r["audio_seconds"] for r in tf))
S["tf_over_capacity"] = block(load("tfatk/over_capacity.jsonl"), ok=lambda r: r["snr_db"] > 30)

fl = collections.defaultdict(list)
fl_rows = [r for r in load("fl/random0.jsonl") if r["name"] in ("local_sgd", "batch")]   # later files replace a stale batch setup
fl_rows += load("fl/random0_optim.jsonl") + load("fl/random0_batch.jsonl")
for r in fl_rows:
    fl[f"{r['name']}|utts={r['n_utts']}|steps={r['steps']}|lr={r['lr']}"].append(r)
S["fl"] = {k: {"n": len(v), "median_mae": st.median(x["mae"] for x in v), "success_rate": sum(x["mae"] < 0.1 for x in v) / len(v),
               "median_total_frames": st.median(x["total_frames"] for x in v)} for k, v in fl.items()}

S["defenses"] = {}
for f in sorted(glob.glob(os.path.join(R, "def/*.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if name.startswith("noise_law"):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["rho"]].append(r)
        S["noise_law"] = {str(k): {"n": len(v), "median_rel": st.median(x["rel"] for x in v),
                                   "median_sigma_over_rms": st.median(x["sigma_over_rms"] for x in v)} for k, v in sorted(by.items())}
    elif rows:
        okf = (lambda r: r["snr_db"] > 30) if name.startswith("tf_") else (lambda r: r["mae"] < 1e-2)
        S["defenses"][name] = block(rows, ok=okf)

S["checkpoints"] = {}
for f in sorted(glob.glob(os.path.join(R, "ckpt/*.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if rows:
        okf = (lambda r: r["snr_db"] > 30) if name.startswith("tf") else (lambda r: r["mae"] < 1e-2)
        S["checkpoints"][name] = block(rows, ok=okf)
S["training"] = {}
for name in ("ds1", "ds1_460", "tf", "tf_460"):
    p = os.path.join(R, name, "log.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        S["training"][name] = [{"step": r["step"], "epoch": r["epoch"], "wer": r["wer"], "cer": r["cer"]} for r in rows]

cl = collections.defaultdict(lambda: collections.defaultdict(list))
caps = {}
for r in load("capacity/cliff.jsonl"):
    cl[r["config"]][r["P"]].append(r["rel"])
    caps[r["config"]] = r["capacity"]
S["capacity_cliff"] = {c: {"predicted": caps[c], "median_rel_by_patches": {str(P): st.median(v) for P, v in sorted(d.items())},
                           "largest_ok": max(P for P, v in d.items() if st.median(v) < 1e-2)} for c, d in cl.items()}
p = os.path.join(R, "capacity/probe.json")
if os.path.exists(p):
    S["architecture_probe"] = json.load(open(p))

S["baseline_gradient_matching"] = []
for f in sorted(glob.glob(os.path.join(R, "baseline/*.jsonl"))):
    for r in [json.loads(l) for l in open(f) if l.strip()]:
        S["baseline_gradient_matching"].append({"config": os.path.basename(f)[:-6], "id": r["id"], "T": r["T"], "mae": r["mae"],
                                                "snr_db": r["snr_db"], "seconds": r["seconds"], "steps": r["steps"],
                                                "final_cosine_distance": r["final_dist"]})
S["conformer"] = {}
for f in sorted(glob.glob(os.path.join(R, "conformer/*.jsonl"))):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if rows:
        S["conformer"][os.path.basename(f)[:-6]] = {
            "n": len(rows), "median_coverage": st.median(r["coverage"] for r in rows),
            "mean_coverage": sum(r["coverage"] for r in rows) / len(rows),
            "full_coverage_rate": sum(r["coverage"] > 0.98 for r in rows) / len(rows),
            "median_snr_db": st.median(r["snr_db"] for r in rows if "snr_db" in r) if any("snr_db" in r for r in rows) else None,
            "median_seconds": st.median(r["seconds"] for r in rows), "median_frames": st.median(r["T"] for r in rows)}
S["audio"] = {}
for f in sorted(glob.glob(os.path.join(R, "audio/*/summary.json"))):
    S["audio"][os.path.basename(os.path.dirname(f))] = json.load(open(f))

os.makedirs("results", exist_ok=True)
json.dump(S, open("results/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k in ("ds1_closed_form", "ds1_sequential", "tf_closed_form", "conformer", "audio")}, indent=1)[:3000])
