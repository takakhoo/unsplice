"""Copy-synthesis check: real audio -> vocoder mel -> HiFi-GAN -> Whisper. Confirms the mel convention."""
import os, sys, torch, soundfile as sf, jiwer, whisper
sys.path.insert(0, ".")
from unsplice.audio import hifigan_mel, load_hifigan
from unsplice.data import list_librispeech

dev = "cuda"
items = list_librispeech(os.environ["LIBRI"], "test-clean")[::130][:20]
voc = load_hifigan(dev, os.environ["HF_HOME"] + "/hifigan")
asr = whisper.load_model("base.en", device=dev)
norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
refs, hyp_real, hyp_voc = [], [], []
for uid, path, text in items:
    w, _ = sf.read(path, dtype="float32")
    wav = torch.from_numpy(w).to(dev)
    with torch.no_grad():
        y = voc.decode_batch(hifigan_mel(wav.unsqueeze(0))).squeeze()
    refs.append(text)
    hyp_real.append(norm(asr.transcribe(wav, language="en")["text"]))
    hyp_voc.append(norm(asr.transcribe(y, language="en")["text"]))
print("WER real audio  :", jiwer.wer(refs, hyp_real))
print("WER copy-synth  :", jiwer.wer(refs, hyp_voc))
print(refs[0]); print(hyp_voc[0])
