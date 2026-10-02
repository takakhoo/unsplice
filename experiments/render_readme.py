"""Fill docs/README.template.md with numbers from results/summary.json and write README.md."""
import json, re

S = json.load(open("results/summary.json"))
T = open("docs/README.template.md").read()
V = {}


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def sci(x):
    return "n/a" if x is None else f"{x:.1e}"


def ck(name):
    return S["checkpoints"].get(name) or {}


def df(name):
    return S["defenses"].get(name) or {}


a, b, c = S["ds1_closed_form"], S["ds1_sequential"], S["tf_closed_form"]
V.update(ds1_n=f"{a['n']:,}", ds1_succ=pct(a["success_rate"]), ds1_snr=f"{a['median_snr_db']:.0f}", ds1_sec=f"{a['median_seconds']:.1f}",
         ds1_fail=a["failures"], ds1_fail_amb=a["failures_flagged_ambiguous"], silence_pct=pct(a["failures"] / a["n"]),
         seq_n=b["n"], seq_succ=pct(b["success_rate"]), seq_snr=f"{b['median_snr_db']:.0f}", seq_sec=f"{b['median_seconds']:.0f}",
         seq_max=f"{b['seconds_of_audio_max']:.1f}", seq_mae=sci(b["median_mae"]),
         tf_n=c["n"], tf_succ=pct(c["success_rate"]), tf_snr=f"{c['median_snr_db']:.0f}", tf_sec=f"{c['median_seconds']:.1f}")
tr = S["training"]
d1 = ck("ds1_ep10_final100h_linear") or ck("ds1_ep8_linear")
d1s = ck("ds1_ep10_final100h_seq")
V.update(ds1ts_n=d1s.get("n", "n/a"), ds1ts_succ=pct(d1s.get("success_rate")), ds1ts_snr=f"{d1s.get('median_snr_db', 0):.0f}")
V.update(ds1t_n=d1.get("n", "n/a"), ds1t_succ=pct(d1.get("success_rate")), ds1t_snr=f"{d1.get('median_snr_db', 0):.0f}",
         ds1_wer=pct(tr["ds1"][-1]["wer"], 0))
dd = df("ds1trained_dropout0.2_linear")
V.update(ds1d_n=dd.get("n", "n/a"), ds1d_succ=pct(dd.get("success_rate")), ds1d_snr=f"{dd.get('median_snr_db', 0):.0f}")
tt, tk = ck("tf_ep26_final100h"), ck("tfknown_ep26_final100h")
V.update(tft_n=tt.get("n", "n/a"), tft_succ=pct(tt.get("success_rate"), 0), tft_snr=f"{tt.get('median_snr_db', 0):.0f}",
         tftk_succ=pct(tk.get("success_rate"), 0), tf_wer=pct(tr["tf"][-1]["wer"], 0))
G = S["baseline_gradient_matching"]
V.update(gm_n=len(G), gm_audio=f"{max(g['T'] for g in G) * 0.02:.1f}", gm_snr=f"{sum(g['snr_db'] for g in G) / len(G):.1f}",
         gm_sec=f"{sum(g['seconds'] for g in G) / len(G):,.0f}", gm_mae=f"{min(g['mae'] for g in G):.1f} to {max(g['mae'] for g in G):.1f}")
A = S["audio"]
main = A.get("ds1_short") or A.get("mfcc_prelim")
V.update(aud_ds1_rec=pct(main["wer"]["rec"]), aud_ds1_orig=pct(main["wer"]["orig"]), aud_ds1_spk=pct(main["speaker_id_top1"], 0),
         aud_ds1_speakers=main.get("enrolled_speakers", main.get("speakers")))
rows = ["| Reconstructions | n | WER original | WER from true features | WER from recovered features | WER Griffin-Lim | Speaker ID, original | Speaker ID, recovered |",
        "|---|---|---|---|---|---|---|---|"]
names = {"ds1_short": "DeepSpeech-1, up to 6.2 s", "ds1_long": "DeepSpeech-1, 6.3 to 35 s", "tf_short": "Transformer-CTC, up to 9.6 s",
         "ds1_trained_dropout": "DeepSpeech-1 trained, dropout 0.2", "tf_trained": "Transformer-CTC trained"}
for key, lab in names.items():
    if key in A:
        r = A[key]
        rows.append(f"| {lab} | {r['n']} | {pct(r['wer']['orig'])}% | {pct(r['wer']['true'])}% | **{pct(r['wer']['rec'])}%** | "
                    f"{pct(r['wer']['gl']) + '%' if 'gl' in r['wer'] else ''} | {pct(r.get('speaker_id_top1_original_audio'), 0)}% | **{pct(r['speaker_id_top1'], 0)}%** |")
V["audio_table"] = "\n".join(rows)
rows = ["| Checkpoint | dev-clean WER | Attack | n | Recovered | Median SNR |", "|---|---|---|---|---|---|"]


def wer_at(run, tag):
    log = tr.get(run, [])
    if not log:
        return ""
    if tag.startswith("step"):
        step = int(tag[4:])
        r = min(log, key=lambda r: abs(r["step"] - step))
    else:
        ep = int("".join(ch for ch in tag.split("_")[0][2:] if ch.isdigit()))
        cand = [r for r in log if r["epoch"] == ep]
        r = cand[-1] if cand else log[-1]
    return pct(r["wer"], 0) + "%"


order = [("ds1_step0_linear", "DeepSpeech-1, random", "ds1", "step0", "closed form"), ("ds1_step1000_linear", "DeepSpeech-1, step 1,000", "ds1", "step1000", "closed form"),
         ("ds1_ep1_linear", "DeepSpeech-1, epoch 1", "ds1", "ep1", "closed form"), ("ds1_ep5_linear", "DeepSpeech-1, epoch 5", "ds1", "ep5", "closed form"),
         ("ds1_ep10_final100h_linear", "DeepSpeech-1, epoch 10 (100 h)", "ds1", "ep10", "closed form"),
         ("ds1_ep1_seq", "DeepSpeech-1, epoch 1", "ds1", "ep1", "sequential, 6.3 to 20 s"), ("ds1_ep5_seq", "DeepSpeech-1, epoch 5", "ds1", "ep5", "sequential, 6.3 to 20 s"),
         ("ds1_ep10_final100h_seq", "DeepSpeech-1, epoch 10 (100 h)", "ds1", "ep10", "sequential, 6.3 to 20 s"),
         ("ds1460_last_linear", "DeepSpeech-1, +1 epoch on 460 h", "ds1_460", "ep1", "closed form"),
         ("ds1460_ep1_seq", "DeepSpeech-1, +1 epoch on 460 h", "ds1_460", "ep1", "sequential, 6.4 to 14 s"),
         ("tf_step0", "Transformer, random", "tf", "step0", "closed form"), ("tf_step3000", "Transformer, step 3,000", "tf", "step3000", "closed form"),
         ("tf_ep5", "Transformer, epoch 5", "tf", "ep5", "closed form"), ("tf_ep26_final100h", "Transformer, epoch 26 (100 h)", "tf", "ep26", "closed form"),
         ("tfknown_ep26_final100h", "Transformer, epoch 26 (100 h)", "tf", "ep26", "closed form, length given"),
         ("tfknown_460_last", "Transformer, +4 epochs on 460 h", "tf_460", "ep4", "closed form, length given"),
         ("tf460_last", "Transformer, +6 epochs on 460 h", "tf_460", "ep6", "closed form")]
for key, lab, run, tag, atk in order:
    r = S["checkpoints"].get(key)
    if r:
        rows.append(f"| {lab} | {wer_at(run, tag)} | {atk} | {r['n']} | {pct(r['success_rate'], 0)}% | {r['median_snr_db']:.0f} dB |")
V["ckpt_table"] = "\n".join(rows)
P = S.get("architecture_probe", {"rows": []})
rows = ["| Model | Layer | Windows in the utterance | Window dimension | Layer width | Gradient rank | Identifies the windows |", "|---|---|---|---|---|---|---|"]
for r in P["rows"]:
    rows.append(f"| {r['model']} | {r['layer']} | {r['patches'] if r['patches'] > 0 else 'many'} | {r['patch_dim']} | {r['width']} | {r['rank']} of {r['max_rank']} | {'yes' if r['span_identifies_patches'] else 'no'} |")
V["probe_table"] = "\n".join(rows)
cl = S["capacity_cliff"]
V["cliff_summary"] = "; ".join(f"{v['largest_ok']} for a prediction of {v['predicted']} ({k.split(' (')[0]})" for k, v in cl.items())
D = S["defenses"]
V["f16_mae"] = sci(df("ds1_f16_known").get("median_mae"))
V["q16_mae"] = sci(df("ds1_quant_16_known").get("median_mae"))
V["prune_succ"] = pct(df("ds1_prune_0.5_known").get("success_rate"), 0)
V["sign_succ"] = pct(df("ds1_sign_known").get("success_rate"), 0)
dr = [(k, v) for k, v in D.items() if k.startswith("ds1_dropout") and k.endswith("linear")]
V["drop_rates"] = ", ".join(k[len("ds1_dropout"):-len("_linear")] for k, _ in dr) if dr else "0.1 to 0.5"
V["drop_succ"] = pct(min(v["success_rate"] for _, v in dr), 0) if dr else "n/a"
singles = {k: v for k, v in S["conformer"].items() if k.endswith("_single")}
if singles:
    key = max(singles, key=lambda k: singles[k]["n"])
    cf = singles[key]
    V.update(conf_cov=pct(cf["mean_coverage"], 0), conf_snr=f"{cf['median_snr_db']:.0f}" if cf["median_snr_db"] else "n/a", conf_n=cf["n"],
             conf_len=f"about {cf['median_frames'] / 100:.0f} s", conf_d=key[1:4])
ds = df("ds1_dropout0.2_seq")
V.update(dropseq_succ=pct(ds.get("success_rate"), 0), dropseq_n=ds.get("n", "n/a"))
V["demo_id"] = json.load(open("docs/audio/index.json"))["featured"]
missing = set(re.findall(r"\{\{(\w+)\}\}", T)) - set(V)
for m in missing:
    V[m] = "n/a"
print("missing placeholders:", sorted(missing))
json.dump({k: v for k, v in V.items() if not k.endswith("_table")}, open("results/readme_values.json", "w"), indent=1)
open("README.md", "w").write(re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), T))
print("wrote README.md")
