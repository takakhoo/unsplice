"""Which front ends leak a usable span? Measure gradient ranks of early layers on a real utterance.

For a layer applying one linear map to P patches of dimension d with C outputs, the gradient
has rank min(P, d, C) at best. The span identifies the patches only while P is below both d
and C, and the closed form needs P <= (kernel - stride) * features on top of that.

Real pretrained weights are used for Whisper and wav2vec 2.0; the other front ends are built
from their published layer shapes (rank depends on shapes, not on training).

Usage: python experiments/capacity_probe.py --out FILE.json
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
sys.path.insert(0, ".")
from unsplice.data import list_librispeech, mfcc, fbank
from unsplice.linear_attack import estimate_rank


def num_rank(g, bias=None):
    M = g.reshape(g.shape[0], -1).double()
    if bias is not None:
        M = torch.cat([M, bias.double().unsqueeze(1)], 1)
    s = torch.linalg.svdvals(torch.nan_to_num(M))
    return int((s > 1e-6 * s[0]).sum()), min(M.shape)   # directions above the float32 noise floor


def row(model, layer, grad, bias, patches, patch_dim, width, closed_form_cap, note=""):
    r, full = num_rank(grad, bias)
    return {"model": model, "layer": layer, "patches": patches, "patch_dim": patch_dim, "width": width,
            "rank": r, "max_rank": full, "span_identifies_patches": bool(r < full and abs(r - patches) <= 3),
            "closed_form_capacity_patches": closed_form_cap, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    items = list_librispeech(os.environ["LIBRI"], "test-clean")
    uid, path, text = min(items, key=lambda it: abs(sf.info(it[1]).frames / 16000 - a.seconds))
    wav, _ = sf.read(path, dtype="float32")
    wav16 = (wav * 32768).astype(np.int16)
    secs = len(wav) / 16000
    rows = []

    # --- wav2vec 2.0 base, fine-tuned with CTC (real weights) ---
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    w2v = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(dev).eval()
    inp = proc(wav, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
    labels = proc.tokenizer(text.upper(), return_tensors="pt").input_ids.to(dev)
    loss = w2v(inp, labels=labels).loss
    names = {"wav2vec2.feature_extractor.conv_layers.0.conv.weight": ("first conv (k=10, s=5, raw audio)", 10, 512, 5),
             "wav2vec2.feature_projection.projection.weight": ("feature projection 512 -> 768", 512, 768, None),
             "wav2vec2.encoder.layers.0.attention.q_proj.weight": ("block 0 attention query", 768, 768, None)}
    params = dict(w2v.named_parameters())
    grads = torch.autograd.grad(loss, [params[n] for n in names], allow_unused=True)
    frames = int(w2v._get_feat_extract_output_lengths(inp.shape[1]))
    n0 = (inp.shape[1] - 10) // 5 + 1
    for (n, (desc, d, width, cap)), g in zip(names.items(), grads):
        P = n0 if "conv_layers.0" in n else frames
        bias_name = n.replace(".weight", ".bias")
        rows.append(row("wav2vec 2.0 base (960h, real weights)", desc, g, None, P, d, width, cap))
    del w2v

    # --- Whisper tiny.en (real weights). Inputs are always padded to 30 s = 3000 frames. ---
    import whisper
    wm = whisper.load_model("tiny.en", device=dev).train()
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(wav))).unsqueeze(0).to(dev)
    tok = whisper.tokenizer.get_tokenizer(multilingual=False)
    ids = torch.tensor([[*tok.sot_sequence_including_notimestamps, *tok.encode(" " + text), tok.eot]], device=dev)
    logits = wm(mel, ids[:, :-1])
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:])
    g1, b1, g2 = torch.autograd.grad(loss, [wm.encoder.conv1.weight, wm.encoder.conv1.bias, wm.encoder.conv2.weight])
    rows.append(row("Whisper tiny.en (real weights)", "encoder conv1 (k=3, s=1, 80 mel)", g1, b1, 3000, 240, 384, 160,
                    "input always padded to 3000 frames"))
    rows.append(row("Whisper tiny.en (real weights)", "encoder conv2 (k=3, s=2, 384 ch)", g2, None, 1500, 1152, 384, None))
    del wm

    # --- shape-determined front ends, random weights ---
    def ctc_head(feat, width, P):
        # a nonlinear stack stands in for the rest of the network, so backward signals are not confined
        # to the 28-dimensional space a bare linear output layer would give them
        head = nn.Sequential(nn.Linear(width, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 29)).to(dev)
        y = torch.randint(1, 29, (max(2, P // 8),), device=dev)
        return nn.functional.ctc_loss(head(feat).log_softmax(-1).unsqueeze(1), y.unsqueeze(0), torch.tensor([P]),
                                      torch.tensor([len(y)]), zero_infinity=True)

    m = torch.from_numpy(mfcc(wav16)).to(dev)                         # (T, 26) at 50 Hz
    fb = torch.from_numpy(fbank(wav16)).to(dev)                        # (T, 80) at 100 Hz
    fb = (fb - fb.mean()) / fb.std()
    # DeepSpeech-1 first layers, context 6 (the earlier attack) and 9 (Mozilla DeepSpeech default)
    from unsplice.model import splice
    for c in (6, 9):
        fc = nn.Linear(26 * (2 * c + 1), 2048).to(dev)
        h = torch.clamp(fc(splice(m.unsqueeze(1), c).squeeze(1)), 0, 20)
        g, b = torch.autograd.grad(ctc_head(h, 2048, len(m)), [fc.weight, fc.bias])
        rows.append(row(f"DeepSpeech-1 (context {c})", f"fc1 on {2*c+1} spliced MFCC frames", g, b, len(m),
                        26 * (2 * c + 1), 2048, 2 * c * 26))
    # pfl4asr Transformer-CTC: Conv1d(80, 1536, k=7, s=3)
    conv = nn.Conv1d(80, 1536, 7, stride=3, padding=3).to(dev)
    h = conv(fb.T.unsqueeze(0)).squeeze(0).T
    h = h[:, :768] * torch.sigmoid(h[:, 768:])                      # GLU, as in the real encoder
    g, b = torch.autograd.grad(ctc_head(h, 768, h.shape[0]), [conv.weight, conv.bias])
    rows.append(row("Transformer-CTC (pfl4asr)", "Conv1d k=7, s=3 on 80 mel", g, b, h.shape[0], 560, 1536, 320))
    # DeepSpeech-2: Conv2d(1, 32, (41, 11), stride (2, 2)) on a 161-bin spectrogram, then RNN input
    spec = torch.stft(torch.from_numpy(wav).to(dev), 320, 160, 320, torch.hamming_window(320, device=dev), return_complex=True).abs()
    spec = torch.log1p(spec).unsqueeze(0).unsqueeze(0)                # (1, 1, 161, T)
    c1 = nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5)).to(dev)
    c2 = nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5)).to(dev)
    f = torch.clamp(c2(torch.clamp(c1(spec), 0, 20)), 0, 20)       # (1, 32, F', T')
    seq = f.permute(3, 0, 1, 2).reshape(f.shape[3], -1)              # (T', 32 * F')
    rnn_in = nn.Linear(seq.shape[1], 800).to(dev)                    # stands in for the first RNN's input weights
    hh = torch.tanh(rnn_in(seq))
    g1, b1, g3, b3 = torch.autograd.grad(ctc_head(hh, 800, hh.shape[0]), [c1.weight, c1.bias, rnn_in.weight, rnn_in.bias])
    rows.append(row("DeepSpeech-2", "Conv2d 41x11, 32 channels", g1, b1, int(c1(spec).shape[2] * c1(spec).shape[3]), 451, 32, None))
    rows.append(row("DeepSpeech-2", "first RNN input weights", g3, b3, hh.shape[0], seq.shape[1], 800, None))
    # QuartzNet: depthwise Conv1d(64, 64, k=33, groups=64, s=2) then pointwise 64 -> 256
    fb64 = fb[:, :64]
    dw = nn.Conv1d(64, 64, 33, stride=2, padding=16, groups=64).to(dev)
    pw = nn.Conv1d(64, 256, 1).to(dev)
    hq = torch.relu(pw(dw(fb64.T.unsqueeze(0)))).squeeze(0).T
    gd, gp, bp = torch.autograd.grad(ctc_head(hq, 256, hq.shape[0]), [dw.weight, pw.weight, pw.bias])
    rows.append(row("QuartzNet", "depthwise Conv1d k=33 (one kernel per mel band)", gd, None, hq.shape[0], 33, 1, None,
                    "each band's kernel gradient is a single vector"))
    rows.append(row("QuartzNet", "pointwise 64 -> 256", gp, bp, hq.shape[0], 64, 256, None))
    # Conformer / ESPnet Conv2dSubsampling: two 3x3 stride-2 convs, then Linear(C * F', d)
    for d in (256, 512):
        s1 = nn.Conv2d(1, d, 3, 2).to(dev)
        s2 = nn.Conv2d(d, d, 3, 2).to(dev)
        z = torch.relu(s2(torch.relu(s1(fb.unsqueeze(0).unsqueeze(0)))))    # (1, d, T', F')
        seq = z.permute(2, 0, 1, 3).reshape(z.shape[2], -1)
        lin = nn.Linear(seq.shape[1], d).to(dev)
        hc = torch.relu(lin(seq))
        g1, b1, gl, bl = torch.autograd.grad(ctc_head(hc, d, hc.shape[0]), [s1.weight, s1.bias, lin.weight, lin.bias])
        rows.append(row(f"Conformer-CTC (Conv2dSubsampling, d={d})", "first Conv2d 3x3", g1, b1, -1, 9, d, None))
        rows.append(row(f"Conformer-CTC (Conv2dSubsampling, d={d})", f"Linear {seq.shape[1]} -> {d} after subsampling", gl, bl,
                        hc.shape[0], seq.shape[1], d, None))
    json.dump({"utterance": uid, "seconds": secs, "rows": rows}, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['model']:45s} {r['layer']:48s} patches={r['patches']:6d} dim={r['patch_dim']:5d} width={r['width']:5d} "
              f"rank={r['rank']:5d}/{r['max_rank']:5d} identifies={r['span_identifies_patches']}")


if __name__ == "__main__":
    main()
