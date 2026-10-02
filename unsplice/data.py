"""LibriSpeech utterances and the DeepSpeech-1 MFCC front end."""
import os
import numpy as np
import soundfile as sf
import torch
import python_speech_features as psf

SR = 16000
WINLEN, WINSTEP, NUMCEP = 0.025, 0.02, 26


def mfcc(audio, numcep=NUMCEP):
    """Same call as the Myrtle DeepSpeech port: 26 cepstra, 25 ms window, 20 ms hop."""
    return psf.mfcc(audio, samplerate=SR, winlen=WINLEN, winstep=WINSTEP, numcep=numcep).astype(np.float32)


def list_librispeech(root, split="test-clean"):
    """Return [(utt_id, flac_path, transcript)] sorted by id."""
    items = []
    base = os.path.join(root, split)
    for spk in sorted(os.listdir(base)):
        for chap in sorted(os.listdir(os.path.join(base, spk))):
            d = os.path.join(base, spk, chap)
            with open(os.path.join(d, f"{spk}-{chap}.trans.txt")) as f:
                for line in f:
                    uid, text = line.strip().split(" ", 1)
                    items.append((uid, os.path.join(d, uid + ".flac"), text.lower()))
    return items


def load_utterance(path):
    audio, sr = sf.read(path, dtype="int16")
    assert sr == SR
    return audio


def features(path):
    """(T, 26) float32 tensor of MFCCs for one file."""
    return torch.from_numpy(mfcc(load_utterance(path)))


def build_index(root, split="test-clean", cache=None):
    """List utterances with their frame counts; cached because it reads every header."""
    if cache and os.path.exists(cache):
        return torch.load(cache)
    out = []
    for uid, path, text in list_librispeech(root, split):
        n = sf.info(path).frames
        T = 1 + int(np.ceil(max(0, n - WINLEN * SR) / (WINSTEP * SR)))
        out.append({"id": uid, "path": path, "text": text, "samples": n, "frames": T})
    if cache:
        torch.save(out, cache)
    return out


def fbank(audio, n_mels=80):
    """80-bin log-mel filterbank, 25 ms window, 10 ms hop (Kaldi convention), float32 (T, n_mels)."""
    import torchaudio
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=n_mels, sample_frequency=SR, dither=0.0,
                                             frame_length=25.0, frame_shift=10.0).numpy()


class FeatureSet:
    """Precomputed features for one split (see experiments/prep_features.py)."""

    def __init__(self, feat_dir, split, kind="mfcc"):
        self.index = torch.load(os.path.join(feat_dir, f"{split}.index.pt"))
        self.data = np.load(os.path.join(feat_dir, f"{split}.{kind}.npy"), mmap_mode="r")
        self.kind = kind

    def __len__(self):
        return len(self.index)

    def frames(self, i):
        return self.index[i][self.kind][1]

    def __getitem__(self, i):
        off, n = self.index[i][self.kind]
        return torch.from_numpy(np.array(self.data[off:off + n], dtype=np.float32)), self.index[i]["text"]


def boundary_windows(fs, n, width, seed=0, last=False):
    """Opening (or closing) `width` frames of n utterances from a public set, used as starting points."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in torch.randperm(len(fs), generator=g).tolist()[:n]:
        x, _ = fs[i]
        out.append(x[-width:] if last else x[:width])
    return torch.stack(out)


def frame_codebook(fs, n=4096, seed=0):
    """n frames drawn from a public set, a few per utterance."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in torch.randperm(len(fs), generator=g).tolist()[:n // 8]:
        x, _ = fs[i]
        out.append(x[torch.randperm(len(x), generator=g)[:8]])
    return torch.cat(out)
