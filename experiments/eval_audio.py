"""Turn reconstructed features into audio and measure what a listener (or a machine) gets from it.

For each saved reconstruction: synthesise audio from the recovered features and from the true
features, transcribe both with Whisper, and compare speaker embeddings with the original
recording. Also writes wav files for the demo page.

Usage: python experiments/eval_audio.py --recon DIR --src mfcc|fbank --f2m CKPT --out DIR [--n 100]
"""
import argparse, glob, json, os, sys
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, ".")
from unsplice.audio import Feat2Mel, load_hifigan, mfcc_to_logmel, logmel_to_wave_griffinlim
from unsplice.data import list_librispeech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--f2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--save-wavs", type=int, default=12)
    ap.add_argument("--whisper", default="small.en")
    ap.add_argument("--enrol", type=int, default=5, help="original utterances per speaker used for enrolment")
    ap.add_argument("--griffinlim", type=int, default=0, help="also run Griffin-Lim on this many utterances (MFCC only)")
    a = ap.parse_args()
    import jiwer, whisper
    from whisper.normalizers import EnglishTextNormalizer
    from speechbrain.inference.speaker import EncoderClassifier
    dev = "cuda"
    os.makedirs(os.path.join(a.out, "wav"), exist_ok=True)
    paths = {u: p for u, p, _ in list_librispeech(os.environ["LIBRI"], "test-clean")}
    f2m = Feat2Mel(26 if a.src == "mfcc" else 80).to(dev).eval()
    f2m.load_state_dict(torch.load(a.f2m, map_location=dev)["model"])
    voc = load_hifigan(dev, os.environ["HF_HOME"] + "/hifigan")
    asr = whisper.load_model(a.whisper, device=dev)
    spk = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=os.environ["HF_HOME"] + "/ecapa",
                                         run_opts={"device": dev})
    norm = EnglishTextNormalizer()
    files = sorted(glob.glob(os.path.join(a.recon, "*.pt")))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.permutation(len(files))[:a.n]]

    @torch.no_grad()
    def synth(feats, n_samples):
        mel = f2m(feats.unsqueeze(0).to(dev), n_samples // 256 + 1)
        return voc.decode_batch(mel).squeeze().cpu()

    @torch.no_grad()
    def embed(wav):
        return torch.nn.functional.normalize(spk.encode_batch(wav.unsqueeze(0).to(dev)).squeeze(), dim=0).cpu()

    def transcribe(wav):
        return norm(asr.transcribe(wav.float().to(dev), language="en", fp16=False)["text"])

    rows, texts = [], {"ref": [], "orig": [], "true": [], "rec": [], "gl": []}
    embs = {}
    for k, fpath in enumerate(files):
        d = torch.load(fpath)
        uid = d["id"]
        w, _ = sf.read(paths[uid], dtype="float32")
        orig = torch.from_numpy(w)
        if d["X"].shape != d["x"].shape:
            continue
        w_true, w_rec = synth(d["x"], len(w)), synth(d["X"], len(w))
        ref = norm(d["text"])
        hyp = {"orig": transcribe(orig), "true": transcribe(w_true), "rec": transcribe(w_rec)}
        e_orig, e_rec, e_true = embed(orig), embed(w_rec), embed(w_true)
        embs[uid] = {"orig": e_orig, "rec": e_rec, "spk": uid.split("-")[0]}
        row = {"id": uid, "seconds": len(w) / 16000, "ref": ref, **{"hyp_" + k2: v for k2, v in hyp.items()},
               "spk_sim_rec": float(e_orig @ e_rec), "spk_sim_true": float(e_orig @ e_true),
               "feat_mae": float((d["X"] - d["x"]).abs().mean())}
        if a.src == "mfcc" and k < a.griffinlim:
            w_gl = torch.from_numpy(logmel_to_wave_griffinlim(mfcc_to_logmel(d["X"].numpy().astype(np.float64))).astype(np.float32))
            row["hyp_gl"] = transcribe(w_gl)
            texts["gl"].append((ref, row["hyp_gl"]))
            if k < a.save_wavs:
                sf.write(os.path.join(a.out, "wav", f"{uid}_griffinlim.wav"), w_gl.numpy(), 16000)
        for name in ("orig", "true", "rec"):
            texts[name].append((ref, hyp[name]))
        if k < a.save_wavs:
            sf.write(os.path.join(a.out, "wav", f"{uid}_original.wav"), orig.numpy(), 16000)
            sf.write(os.path.join(a.out, "wav", f"{uid}_reconstructed.wav"), w_rec.numpy(), 16000)
            sf.write(os.path.join(a.out, "wav", f"{uid}_true_features.wav"), w_true.numpy(), 16000)
        rows.append(row)
    # Speaker identification: enrol every test-clean speaker from original recordings that were not attacked,
    # then assign each reconstruction to the nearest speaker centroid.
    attacked = set(embs)
    by_spk = {}
    for uid, path in sorted(paths.items()):
        if uid not in attacked:
            by_spk.setdefault(uid.split("-")[0], []).append(path)
    cents = {}
    for spk_id, plist in by_spk.items():
        es = []
        for path in plist[:a.enrol]:
            w, _ = sf.read(path, dtype="float32")
            es.append(embed(torch.from_numpy(w)))
        cents[spk_id] = torch.nn.functional.normalize(torch.stack(es).mean(0), dim=0)
    names = sorted(cents)
    C = torch.stack([cents[n] for n in names])
    correct = {"rec": 0, "orig": 0}
    n_enrol = 0
    for uid, e in embs.items():
        if e["spk"] not in cents:
            continue
        n_enrol += 1
        for key in correct:
            correct[key] += names[int((C @ e[key]).argmax())] == e["spk"]
    summary = {"n": len(rows), "whisper": a.whisper,
               "wer": {k: jiwer.wer([r for r, _ in v], [h for _, h in v]) for k, v in texts.items() if v},
               "spk_sim_rec_mean": float(np.mean([r["spk_sim_rec"] for r in rows])),
               "spk_sim_true_mean": float(np.mean([r["spk_sim_true"] for r in rows])),
               "speaker_id_top1": correct["rec"] / max(1, n_enrol), "speaker_id_top1_original_audio": correct["orig"] / max(1, n_enrol),
               "enrolled_speakers": len(names), "enrol_utterances": a.enrol, "n_scored": n_enrol}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "per_utterance.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
