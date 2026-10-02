"""Draw every figure from results/summary.json and results/raw. Usage: python experiments/make_figures.py"""
import collections, glob, json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R, OUT = "results/raw", "docs/figures"
os.makedirs(OUT, exist_ok=True)
S = json.load(open("results/summary.json"))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(R, pat))) for l in open(f) if l.strip()]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_overview():
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.axis("off")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 35)

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="white", ec=color, lw=1.4))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.45)

    def arrow(x0, x1, y, label=None):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=MUTED))
        if label:
            ax.text((x0 + x1) / 2, y + 1.3, label, ha="center", fontsize=7.5, color=MUTED)

    box(0.5, 9, 17, 19, "Client (private)", "utterance of T frames\nfeatures X (T x F)\nwindows $c_t$ of k frames\none training step", BLUE)
    box(22.5, 9, 20, 19, "Server receives", "$\\nabla W_1=\\sum_t g_t\\,c_t^{\\top}$\n$\\nabla b_1=\\sum_t g_t$\nno audio, no transcript", ORANGE)
    box(47.5, 9, 19, 19, "Row space", "SVD of [$\\nabla W_1$ | $\\nabla b_1$]\nrank = number of frames\nspan{[$c_t$, 1]} is known", AQUA)
    box(71.5, 9, 19, 19, "Linear system", "windows overlap, so\n[$c_t(X)$, 1] in span\nis linear in X;\nunique if T ≤ (k−s)F", AQUA)
    box(95.5, 9, 9, 19, "Output", "X exactly:\naudio,\nwords,\nspeaker", BLUE)
    arrow(18.1, 21.9, 18.5, "update")
    arrow(43.1, 46.9, 18.5)
    arrow(67.1, 70.9, 18.5)
    arrow(91.1, 94.9, 18.5)
    ax.text(52.5, 31.5, "Closed-form recovery of speech features from one federated update", ha="center", fontsize=11.5, fontweight="bold")
    ax.text(52.5, 3.5, "Past the first-layer limit, deeper layers leak span{[$h(c_t)$, 1]} in 2,048 to 4,096 dimensions; frames are then decoded one at a time, each checked against the span.",
            ha="center", fontsize=8, color=INK2)
    save(fig, "overview")


def fig_cliff():
    cl = collections.defaultdict(lambda: collections.defaultdict(list))
    caps = {}
    for r in load("capacity/cliff.jsonl"):
        cl[r["config"]][r["P"]].append(r["rel"])
        caps[r["config"]] = r["capacity"]
    fig, axes = plt.subplots(1, len(cl), figsize=(3.0 * len(cl), 2.9), sharey=True)
    for ax, (cfg, d), col in zip(np.atleast_1d(axes), cl.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        cap = caps[cfg]
        P = sorted(p for p in d if abs(p - cap) <= 24)
        for p in P:
            ax.scatter([p - cap] * len(d[p]), d[p], s=9, color=col, alpha=0.45, linewidths=0)
        ax.plot([p - cap for p in P], [st.median(d[p]) for p in P], color=col, lw=2)
        ax.axvline(0.5, color=INK2, lw=0.9, ls=(0, (3, 3)))
        ax.set_yscale("log")
        ax.set_title(cfg.split(" (")[0], fontsize=9)
        ax.set_xlabel(f"patches − {cap}")
        ax.text(0.04, 0.5, cfg.split(" (")[1].rstrip(")"), transform=ax.transAxes, fontsize=7.5, color=MUTED)
    np.atleast_1d(axes)[0].set_ylabel("relative feature error")
    fig.suptitle("Error of the closed form around the predicted capacity (k−s)F (dashed)", y=1.03, fontsize=10.5, fontweight="bold")
    save(fig, "capacity_cliff")


def fig_examples():
    d = np.load("results/figdata/examples.npz", allow_pickle=True)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [1, 3.2]})
    for col, tag in enumerate(("short", "long")):
        x, X = d[tag + "_x"], d[tag + "_X"]
        vmin, vmax = np.percentile(x[:, 1:], [1, 99])
        for row, (arr, name) in enumerate(((x, "true MFCC"), (X, "recovered from the gradient"))):
            ax = axes[row, col]
            ax.imshow(arr[:, 1:].T, aspect="auto", origin="lower", cmap="Blues", vmin=vmin, vmax=vmax,
                      extent=[0, len(arr) * 0.02, 1, 26])
            ax.grid(False)
            ax.set_ylabel("cepstral index")
            ax.set_title(f"recovered, {'closed form' if tag == 'short' else 'sequential decoding'}" if row else f"true MFCC ({len(arr) * 0.02:.1f} s)", fontsize=9, loc="left")
            ax.set_xticklabels([])
        ax = axes[2, col]
        err = np.abs(X - x).mean(1)
        ax.semilogy(np.arange(len(err)) * 0.02, err, color=ORANGE, lw=1.2)
        ax.set_ylim(1e-8, 1e1)
        ax.axhline(np.abs(x).mean(), color=MUTED, lw=0.9, ls=(0, (3, 3)))
        ax.text(0.01, np.abs(x).mean() * 1.6, "mean |feature|", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, len(err) * 0.02)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("abs. error per frame")
        ax.set_title(f"error (mean {err.mean():.1e})", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "examples")


def fig_spectrum():
    sp = json.load(open("results/figdata/spectra.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
    ax = axes[0]
    for (uid, e), col in zip([(u, e) for u, e in sp.items() if e["T"] < 320], [BLUE, ORANGE]):
        s = np.array(e["fc1"])
        ax.semilogy(np.arange(1, len(s) + 1), s, color=col, lw=2, label=f"{e['T']} frames ({e['T'] * 0.02:.1f} s)")
        ax.axvline(e["T"] + 0.5, color=col, lw=0.9, ls=(0, (3, 3)))
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value / largest")
    ax.set_title("First layer: the rank is the frame count")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    e = [e for e in sp.values() if e["T"] >= 900][0]
    for key, col, name in (("fc2", BLUE, "fc2 (inputs in 2,048 dims)"), ("lstm", AQUA, "LSTM input (4,096 dims)"), ("fc1", ORANGE, "fc1 (338 dims, saturated)")):
        s = np.array(e[key])
        ax.semilogy(np.arange(1, len(s) + 1), s, color=col, lw=2, label=name)
    ax.axvline(e["T"] + 0.5, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.text(e["T"] + 20, 3e-2, f"T = {e['T']}", fontsize=8, color=INK2)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("singular value index")
    ax.set_title(f"A {e['T'] * 0.02:.0f} s utterance: deeper layers still show the gap")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    save(fig, "spectrum")


def fig_duration():
    e1, e2 = load("e1/random0_linear.*.jsonl"), (load("e2b/*.jsonl") or load("e2/*.jsonl"))
    gm = load("baseline/*.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    ax = axes[0]
    ok1 = [r for r in e1 if (r.get("ambiguity") or 0) <= 1e-2 or r["mae"] < 1e-2]
    amb = [r for r in e1 + e2 if r["mae"] >= 1e-2]
    ax.scatter([r["audio_seconds"] for r in ok1], [r["mae"] for r in ok1], s=5, color=BLUE, alpha=0.5, linewidths=0, label="closed form (first layer)")
    ax.scatter([r["audio_seconds"] for r in e2 if r["mae"] < 1e-2], [r["mae"] for r in e2 if r["mae"] < 1e-2], s=9, color=AQUA, alpha=0.8, linewidths=0, label="sequential (deeper layers)")
    ax.scatter([r["audio_seconds"] for r in amb], [r["mae"] for r in amb], s=14, color=MAGENTA, marker="x", linewidths=1, label="digital silence (ambiguous)")
    if gm:
        ax.scatter([r["T"] * 0.02 for r in gm], [r["mae"] for r in gm], s=40, color=ORANGE, marker="D", label="gradient matching baseline")
    ax.scatter([10], [8.78], s=45, color=YELLOW, marker="s", label="earlier paper, 10 s grid run")
    ax.set_yscale("log")
    ax.axvline(6.24, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.set_ylim(1e-7, 30)
    ax.text(6.6, 2e-7, "first-layer limit, 312 frames", fontsize=7.5, color=INK2)
    ax.set_xlabel("utterance length (s)")
    ax.set_ylabel("MFCC mean absolute error")
    ax.set_title("Reconstruction error, one gradient per utterance")
    ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.07))
    ax = axes[1]
    ax.scatter([r["audio_seconds"] for r in ok1], [r["seconds"] for r in ok1], s=5, color=BLUE, alpha=0.5, linewidths=0)
    ax.scatter([r["audio_seconds"] for r in e2], [r["seconds"] for r in e2], s=9, color=AQUA, alpha=0.8, linewidths=0)
    if gm:
        ax.scatter([r["T"] * 0.02 for r in gm], [r["seconds"] for r in gm], s=40, color=ORANGE, marker="D")
    ax.scatter([10], [3200], s=45, color=YELLOW, marker="s")
    ax.set_yscale("log")
    ax.set_xlabel("utterance length (s)")
    ax.set_ylabel("attack time (s, shared GPU)")
    ax.set_title("Compute")
    fig.tight_layout()
    save(fig, "duration")


def fig_noise():
    nl = load("def/noise_law*.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax = axes[0]
    ax.scatter([r["rho"] for r in nl], [r["rel"] for r in nl], s=8, color=BLUE, alpha=0.35, linewidths=0)
    by = collections.defaultdict(list)
    for r in nl:
        by[r["rho"]].append(r["rel"])
    ks = sorted(by)
    ax.plot(ks, [st.median(by[k]) for k in ks], color=BLUE, lw=2, label="median over utterances")
    ax.plot([1e-4, 1], [1e-4 * 0.5, 0.5], color=INK2, lw=0.9, ls=(0, (3, 3)), label="slope 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$ = noise spectral norm / weakest signal singular value")
    ax.set_ylabel("relative feature error")
    ax.set_title("Noise hides the span when it reaches the weakest direction")
    ax.legend(fontsize=8)
    ax = axes[1]
    D = S["defenses"]
    levels = ["1e-7", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]
    vals = [D.get(f"ds1_noise_{l}_known", {}).get("median_mae") for l in levels]
    ax.plot([float(l) for l, v in zip(levels, vals) if v], [v for v in vals if v], color=ORANGE, lw=2, marker="o", ms=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gaussian noise std / RMS of the gradient tensor")
    ax.set_ylabel("median MFCC MAE")
    ax.set_title("DeepSpeech-1, utterances up to 6.2 s")
    fig.tight_layout()
    save(fig, "noise")


def fig_training():
    tr = S["training"]

    def series(prefix, runs, skip=("seq", "recon")):
        """(global step, success, wer, n) for checkpoints whose result name starts with prefix + '_' or prefix + '460_'."""
        rows = []
        for name, blk in S["checkpoints"].items():
            head, _, tag = name.partition("_")
            if head not in (prefix, prefix + "460") or any(k in name for k in skip):
                continue
            run = runs[1] if head.endswith("460") else runs[0]
            log = tr.get(run, [])
            if not log:
                continue
            offset = tr[runs[0]][-1]["step"] if run == runs[1] else 0
            tag = tag.split("_")[0]
            if tag.startswith("step"):
                step = int(tag[4:])
            elif tag == "last":      # checkpoint as it stood when attacked: epoch 1 (DeepSpeech-1) and epoch 6 (Transformer)
                ep = {"ds1460": 1, "tf460": 6}[head]
                step = max(r["step"] for r in log if r["epoch"] == ep)
            else:
                ep = int("".join(ch for ch in tag[2:] if ch.isdigit()))
                cand = [r["step"] for r in log if r["epoch"] == ep]
                if not cand:
                    continue
                step = max(cand)
            wer = min(log, key=lambda r: abs(r["step"] - step))["wer"]
            rows.append((max(step + offset, 30), blk["success_rate"], wer, blk["n"]))
        return sorted(rows)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.0), sharex="col")
    for col, (title, runs, sets, thr) in enumerate((
            ("DeepSpeech-1", ["ds1", "ds1_460"], [("ds1", BLUE, "closed form")], "MAE < 0.01"),
            ("Transformer-CTC", ["tf", "tf_460"], [("tf", BLUE, "length not given"), ("tfknown", AQUA, "length given")], "SNR > 30 dB"))):
        logs = [(r["step"], r["wer"]) for r in tr.get(runs[0], [])]
        if runs[1] in tr and logs:
            logs += [(logs[-1][0] + r["step"], r["wer"]) for r in tr[runs[1]] if r["step"] > 0]
        ax = axes[0, col]
        ax.plot([max(x, 30) for x, _ in logs], [100 * w for _, w in logs], color=ORANGE, lw=2)
        ax.set_ylabel("dev-clean WER (%)")
        ax.set_title(f"{title}: word error rate along training")
        ax.set_xscale("log")
        ax = axes[1, col]
        for prefix, colr, lab in sets:
            rows = series(prefix, runs)
            if rows:
                ax.plot([r[0] for r in rows], [100 * r[1] for r in rows], color=colr, lw=2, marker="o", ms=4, label=lab)
        ax.set_ylim(0, 105)
        ax.set_ylabel(f"utterances recovered (%)\n[{thr}]")
        ax.set_xlabel("training step of the attacked checkpoint")
        ax.set_title("Attack success at each checkpoint")
        if len(sets) > 1:
            ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "training")


def fig_fl():
    F = S["fl"]
    order = [("local_sgd|utts=1|steps=1|lr=0.01", "1 SGD step"), ("local_sgd|utts=1|steps=10|lr=0.01", "10 SGD steps"),
             ("local_sgd|utts=1|steps=50|lr=0.01", "50 SGD steps"), ("sgd_momentum|utts=1|steps=10|lr=0.1", "SGD + momentum 0.9"),
             ("sgd_weight_decay|utts=1|steps=10|lr=0.1", "SGD + weight decay"), ("batch|utts=2|steps=1|lr=0.1", "batch of 2 utterances"),
             ("batch_short|utts=4|steps=1|lr=0.1", "batch of 4 short"), ("batch_short|utts=8|steps=1|lr=0.1", "batch of 8 short"),
             ("batch_short|utts=16|steps=1|lr=0.1", "batch of 16 short"), ("local_epochs_batch1|utts=8|steps=16|lr=0.1", "8 utterances, 2 local epochs"),
             ("over_budget|utts=4|steps=1|lr=0.1", "over capacity (4 x 1.8 s)"), ("over_budget|utts=8|steps=1|lr=0.1", "over capacity (8 x 1.2 s)"),
             ("adam|utts=1|steps=1|lr=0.001", "1 Adam step"), ("adam|utts=1|steps=10|lr=0.001", "10 Adam steps")]
    rows = [(lab, F[k]) for k, lab in order if k in F]
    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(rows) + 1.1))
    y = np.arange(len(rows))[::-1]
    vals = [r[1]["median_mae"] for r in rows]
    ax.barh(y, vals, height=0.62, color=[BLUE if v < 0.1 else MUTED for v in vals])
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK2)
    for yi, v, r in zip(y, vals, rows):
        ax.text(v * 1.25, yi, f"{v:.1e}  ({int(r[1]['median_total_frames'])} frames)", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(1e-5, 3e3)
    ax.axvline(0.1, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("median MFCC MAE of the recovered features (features have mean magnitude about 8)")
    ax.set_title("What the server receives in practice: closed-form attack on the first-layer update", loc="left")
    save(fig, "fl_surfaces")


def fig_audio():
    A = S.get("audio", {})
    if not A:
        return
    names = [k for k in A]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax = axes[0]
    labels = [("orig", "original\nrecording"), ("true", "vocoded\ntrue features"), ("rec", "vocoded\nrecovered features"), ("gl", "Griffin-Lim\nrecovered")]
    w = 0.8 / len(names)
    for j, (n, col) in enumerate(zip(names, [BLUE, ORANGE, AQUA])):
        vals = [100 * A[n]["wer"].get(k, np.nan) for k, _ in labels]
        ax.bar(np.arange(len(labels)) + j * w - 0.4 + w / 2, vals, width=w * 0.9, color=col, label=n)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i + j * w - 0.4 + w / 2, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l for _, l in labels], color=INK2, fontsize=8)
    ax.set_ylabel("Whisper WER (%)")
    ax.set_title("An attacker can read the words")
    ax.grid(axis="x", visible=False)
    if len(names) > 1:
        ax.legend(fontsize=7.5)
    ax = axes[1]
    for j, (n, col) in enumerate(zip(names, [BLUE, ORANGE, AQUA])):
        vals = [100 * A[n].get("speaker_id_top1_original_audio", np.nan), 100 * A[n]["speaker_id_top1"]]
        ax.bar(np.arange(2) + j * w - 0.4 + w / 2, vals, width=w * 0.9, color=col)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i + j * w - 0.4 + w / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["original recording", "audio from recovered features"], color=INK2, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("speaker identification top-1 (%)")
    ax.set_title("and tell who is speaking")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "audio")


def fig_paper():
    """Two compact figures sized for a single-column 12 cm text block, readable in grey."""
    plt.rcParams.update({"font.size": 8})
    cl = collections.defaultdict(lambda: collections.defaultdict(list))
    caps = {}
    for r in load("capacity/cliff.jsonl"):
        cl[r["config"]][r["P"]].append(r["rel"])
        caps[r["config"]] = r["capacity"]
    e1, e2, gm = load("e1/random0_linear.*.jsonl"), (load("e2b/*.jsonl") or load("e2/*.jsonl")), load("baseline/*.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35))
    ax = axes[0]
    for (cfg, d), col, mk in zip([(c, d) for c, d in cl.items() if "context 6" in c or "Transformer" in c], [BLUE, ORANGE], ["o", "s"]):
        cap = caps[cfg]
        P = sorted(p for p in d if abs(p - cap) <= 24)
        ax.plot([p - cap for p in P], [st.median(d[p]) for p in P], color=col, lw=1.6, marker=mk, ms=3,
                label=("DeepSpeech-1, $(k{-}s)F=312$" if "context 6" in cfg else "Transformer conv, $(k{-}s)F=320$"))
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.set_yscale("log")
    ax.set_xlabel("windows $-\\,(k-s)F$")
    ax.set_ylabel("relative feature error")
    ax.legend(fontsize=6.5, loc="center left")
    ax = axes[1]
    ok1 = [r for r in e1 if r["mae"] < 1e-2]
    ax.scatter([r["audio_seconds"] for r in ok1], [r["mae"] for r in ok1], s=3, color=BLUE, alpha=0.5, linewidths=0, label="closed form")
    ok2 = [r for r in e2 if r["mae"] < 1e-2]
    ax.scatter([r["audio_seconds"] for r in ok2], [r["mae"] for r in ok2], s=7, color=AQUA, marker="^", linewidths=0, label="sequential")
    bad = [r for r in e1 + e2 if r["mae"] >= 1e-2]
    ax.scatter([r["audio_seconds"] for r in bad], [r["mae"] for r in bad], s=10, color=MAGENTA, marker="x", linewidths=0.8, label="digital silence")
    if gm:
        ax.scatter([r["T"] * 0.02 for r in gm], [r["mae"] for r in gm], s=22, color=ORANGE, marker="D", label="gradient matching")
    ax.set_yscale("log")
    ax.set_xlabel("utterance length (s)")
    ax.set_ylabel("MFCC mean abs. error")
    ax.legend(fontsize=6.5, loc="center right")
    fig.tight_layout()
    save(fig, "paper_main")


for f in (fig_paper, fig_overview, fig_cliff, fig_examples, fig_spectrum, fig_duration, fig_noise, fig_training, fig_fl, fig_audio):
    try:
        f()
    except Exception as e:      # a figure whose experiment has not finished yet
        print("skipped", f.__name__, repr(e))
