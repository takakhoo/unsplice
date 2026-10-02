"""Write paper/numbers.tex: one LaTeX macro per number quoted in the papers, read from results/summary.json."""
import json

S = json.load(open("results/summary.json"))
out = []
seen = set()


def put(name, value):
    name = "".join(ch for ch in name.translate(str.maketrans("0123456789", "abcdefghij")) if ch.isalpha())
    if name not in seen:
        seen.add(name)
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def sci(x, digits=1):
    """1.4e-05 -> 1.4 \\times 10^{-5} (math mode)."""
    if x is None:
        return "n/a"
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def pct(x, digits=1):
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


A = S["ds1_closed_form"]
put("DSshortN", f"{A['n']:,}")
put("DSshortSucc", pct(A["success_rate"]))
put("DSshortSNR", f"{A['median_snr_db']:.0f}")
put("DSshortMAE", sci(A["median_mae"]))
put("DSshortSec", f"{A['median_seconds']:.1f}")
put("DSshortFail", A["failures"])
put("DSshortFailAmb", A["failures_flagged_ambiguous"])
B = S["ds1_sequential"]
put("DSlongN", B["n"])
put("DSlongSucc", pct(B["success_rate"]))
put("DSlongMAE", sci(B["median_mae"]))
put("DSlongSNR", f"{B['median_snr_db']:.0f}")
put("DSlongSec", f"{B['median_seconds']:.0f}")
put("DSlongMaxAudio", f"{B['seconds_of_audio_max']:.1f}")
put("DSlongestMAE", sci(B["longest"]["mae"]))
put("DSlongestFrames", f"{B['longest']['T']:,}")
C = S["tf_closed_form"]
put("TFN", C["n"])
put("TFSucc", pct(C["success_rate"]))
put("TFSNR", f"{C['median_snr_db']:.0f}")
put("TFSec", f"{C['median_seconds']:.1f}")
put("TFMaxAudio", f"{C['seconds_of_audio_max']:.1f}")
G = S["baseline_gradient_matching"]
if G:
    put("GMmaeLo", f"{min(g['mae'] for g in G):.1f}")
    put("GMmaeHi", f"{max(g['mae'] for g in G):.1f}")
    put("GMsec", f"{sum(g['seconds'] for g in G) / len(G):,.0f}")
    put("GMn", len(G))
    put("GMaudio", f"{max(g['T'] for g in G) * 0.02:.1f}")
for key, name in (("mfcc_final", "Mfcc"), ("fbank_final", "Fbank"), ("mfcc_prelim", "MfccPrelim")):
    a = S["audio"].get(key)
    if a:
        put(f"Aud{name}N", a["n"])
        for k, lab in (("orig", "Orig"), ("true", "True"), ("rec", "Rec"), ("gl", "GL")):
            if k in a["wer"]:
                put(f"Aud{name}WER{lab}", pct(a["wer"][k]))
        put(f"Aud{name}Spk", pct(a["speaker_id_top1"], 0))
        if "speaker_id_top1_original_audio" in a:
            put(f"Aud{name}SpkOrig", pct(a["speaker_id_top1_original_audio"], 0))
        put(f"Aud{name}Sim", f"{a['spk_sim_rec_mean']:.2f}")
        put(f"Aud{name}Speakers", a.get("enrolled_speakers", a.get("speakers")))
for k, v in S["checkpoints"].items():
    tag = "".join(ch for ch in k.replace("_", " ").title().replace(" ", "") if ch.isalpha() or ch.isdigit())
    tag = tag.translate(str.maketrans("0123456789", "abcdefghij"))
    put("Ck" + tag + "Succ", pct(v["success_rate"], 0))
    put("Ck" + tag + "SNR", f"{v['median_snr_db']:.0f}")
    put("Ck" + tag + "N", v["n"])
for k, v in S["defenses"].items():
    tag = "".join(ch for ch in k.replace("_", " ").replace(".", "p").replace("-", "m").title().replace(" ", "") if ch.isalnum())
    tag = tag.translate(str.maketrans("0123456789", "abcdefghij"))
    put("Df" + tag + "Succ", pct(v["success_rate"], 0))
    put("Df" + tag + "MAE", sci(v["median_mae"]))
for k, v in S["fl"].items():
    tag = "".join(ch for ch in k.replace("|", " ").replace("=", " ").replace(".", "p").replace("_", " ").title().replace(" ", "") if ch.isalnum())
    tag = tag.translate(str.maketrans("0123456789", "abcdefghij"))
    put("Fl" + tag + "MAE", sci(v["median_mae"]))
    put("Fl" + tag + "Succ", pct(v["success_rate"], 0))
for k, v in S["conformer"].items():
    put("Cf" + k.translate(str.maketrans("0123456789", "abcdefghij")) + "Cov", pct(v["mean_coverage"], 0))
    put("Cf" + k.translate(str.maketrans("0123456789", "abcdefghij")) + "SNR", f"{v['median_snr_db']:.0f}" if v["median_snr_db"] else "n/a")
    put("Cf" + k.translate(str.maketrans("0123456789", "abcdefghij")) + "N", v["n"])
for name, rows in S["training"].items():
    tag = name.replace("_", "").translate(str.maketrans("0123456789", "abcdefghij"))
    put("Tr" + tag + "WER", pct(rows[-1]["wer"]))
# Short semantic names for the values the README also quotes.
V = json.load(open("results/readme_values.json"))
names = {"ds1t_succ": "TrDSsucc", "ds1_wer": "TrDSwer", "ds1d_succ": "DropSucc", "tft_succ": "TrTFsucc", "tftk_succ": "TrTFknownSucc",
         "tf_wer": "TrTFwer", "aud_ds1_rec": "WERrec", "aud_ds1_orig": "WERorig", "aud_ds1_spk": "SpkRec", "aud_ds1_speakers": "Speakers",
         "prune_succ": "PruneSucc", "sign_succ": "SignSucc", "conf_cov": "ConfCov", "conf_snr": "ConfSNR", "conf_n": "ConfN",
         "gm_mae": "GMmae", "silence_pct": "SilencePct", "drop_succ": "DropAllSucc"}
for k, n in names.items():
    put(n, V.get(k, "n/a"))
A = S["audio"]
for key, tag in (("ds1_short", "Short"), ("ds1_long", "Long"), ("tf_short", "Tf"), ("ds1_trained_dropout", "Drop"), ("tf_trained", "TfTr")):
    if key in A:
        put("Wer" + tag, pct(A[key]["wer"]["rec"]))
        put("WerOrig" + tag, pct(A[key]["wer"]["orig"]))
        put("Spk" + tag, pct(A[key]["speaker_id_top1"], 0))
        put("SpkOrig" + tag, pct(A[key].get("speaker_id_top1_original_audio"), 0))
        put("AudN" + tag, A[key]["n"])
        if "gl" in A[key]["wer"]:
            put("WerGL" + tag, pct(A[key]["wer"]["gl"]))
for name in ("WerShort", "WerOrigShort", "SpkShort", "SpkOrigShort", "WerLong", "SpkLong", "WerTf", "SpkTf", "WerGLShort", "ConfCov", "ConfSNR",
             "ConfN", "TrTFknownSucc", "TrTFsucc", "WerDrop", "SpkDrop", "WerTfTr", "SpkTfTr", "AudNShort", "AudNLong", "AudNTf", "WerOrigLong", "SpkOrigLong", "WerOrigTf", "SpkOrigTf"):
    put(name, "n/a")          # experiments still running leave a visible gap, not a broken build
open("paper/numbers.tex", "w").write("% Generated by experiments/make_numbers.py from results/summary.json. Do not edit.\n" + "\n".join(out) + "\n")
print(len(out), "macros")

# ---- LaTeX tables ----
import os
os.makedirs("paper/tables", exist_ok=True)


def tex_sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\cdot10^{{{int(e)}}}$"


rows = []
fl_order = [("local_sgd|utts=1|steps=1|lr=0.01", "1 SGD step"), ("local_sgd|utts=1|steps=10|lr=0.01", "10 SGD steps"),
            ("local_sgd|utts=1|steps=50|lr=0.01", "50 SGD steps"), ("sgd_momentum|utts=1|steps=10|lr=0.1", "SGD, momentum 0.9"),
            ("sgd_weight_decay|utts=1|steps=10|lr=0.1", "SGD, weight decay"), ("batch|utts=2|steps=1|lr=0.1", "batch of 2"),
            ("batch_short|utts=8|steps=1|lr=0.1", "batch of 8 (short)"), ("batch_short|utts=16|steps=1|lr=0.1", "batch of 16 (short)"),
            ("over_budget|utts=4|steps=1|lr=0.1", "batch over capacity"), ("adam|utts=1|steps=10|lr=0.001", "10 Adam steps")]
for k, lab in fl_order:
    v = S["fl"].get(k)
    if v:
        rows.append(f"{lab} & {tex_sci(v['median_mae'])} & {100 * v['success_rate']:.0f}\\\\")
def_order = [("ds1_dropout0.2_linear", "dropout 0.2"), ("ds1_dropout0.5_linear", "dropout 0.5"), ("ds1_clip_0.01_blind", "clipping to 0.01"),
             ("ds1_f16_blind", "half precision"), ("ds1_noise_1e-5_known", "noise $10^{-5}$ of RMS"), ("ds1_noise_1e-3_known", "noise $10^{-3}$ of RMS"),
             ("ds1_quant_8_known", "8-bit quantisation"), ("ds1_prune_0.5_known", "prune 50\\%"), ("ds1_sign_known", "sign only")]
rows.append("\\midrule")
for k, lab in def_order:
    v = S["defenses"].get(k)
    if v:
        rows.append(f"{lab} & {tex_sci(v['median_mae'])} & {100 * v['success_rate']:.0f}\\\\")
open("paper/tables/fl.tex", "w").write("\\begin{tabular}{@{}lrr@{}}\n\\toprule\nCondition & Median MAE & Recov.\\ (\\%)\\\\\n\\midrule\n"
                                       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote paper/tables/fl.tex with", len(rows) - 1, "rows")

P = S.get("architecture_probe", {}).get("rows", [])
short = {"wav2vec 2.0 base (960h, real weights)": "wav2vec 2.0", "Whisper tiny.en (real weights)": "Whisper", "DeepSpeech-1 (context 6)": "DeepSpeech-1",
         "Transformer-CTC (pfl4asr)": "Transf.-CTC", "DeepSpeech-2": "DeepSpeech-2", "QuartzNet": "QuartzNet",
         "Conformer-CTC (Conv2dSubsampling, d=256)": "Conformer"}
lay = {"first conv (k=10, s=5, raw audio)": "first conv", "feature projection 512 -> 768": "feature proj.", "encoder conv1 (k=3, s=1, 80 mel)": "conv1",
       "fc1 on 13 spliced MFCC frames": "fc1", "Conv1d k=7, s=3 on 80 mel": "conv", "Conv2d 41x11, 32 channels": "first conv",
       "first RNN input weights": "RNN input", "depthwise Conv1d k=33 (one kernel per mel band)": "depthwise conv", "pointwise 64 -> 256": "pointwise conv",
       "first Conv2d 3x3": "first conv", "Linear 4864 -> 256 after subsampling": "proj. after subs."}
rows = []
for r in P:
    if r["model"] in short and r["layer"] in lay:
        rows.append(f"{short[r['model']]} & {lay[r['layer']]} & {r['patches'] if r['patches'] > 0 else '--'} & {r['rank']}/{r['max_rank']} & {'yes' if r['span_identifies_patches'] else 'no'}\\\\")
open("paper/tables/probe.tex", "w").write("\\begin{tabular}{@{}llrrl@{}}\n\\toprule\nModel & Layer & Windows & Rank & Leaks\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("wrote paper/tables/probe.tex with", len(rows), "rows")
