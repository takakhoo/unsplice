"""Turning recovered features back into sound.

Two routes: a signal-processing inverse of the DeepSpeech MFCC (no learning, robotic but
honest), and a small network that maps features to the mel spectrogram of a pretrained
HiFi-GAN vocoder (natural-sounding; pitch is inferred because 26 mel bands do not carry it).
"""
import math
import numpy as np
import torch
import torch.nn as nn

SR = 16000
N_FFT, WIN, HOP, N_FILT, LIFTER, PREEMPH = 512, 400, 320, 26, 22, 0.97


def _mel_filterbank():
    import python_speech_features as psf
    return psf.get_filterbanks(N_FILT, N_FFT, SR, 0, SR / 2)          # (26, 257)


def mfcc_to_logmel(mfcc):
    """Invert the cepstral steps of python_speech_features.mfcc. mfcc: (T, 26) numpy -> (T, 26) log energies.

    The first coefficient was replaced by log frame energy, so c0 is restored by requiring the
    filterbank energies to add up to that total.
    """
    from scipy.fftpack import idct
    n = np.arange(mfcc.shape[1])
    lift = 1 + (LIFTER / 2.0) * np.sin(np.pi * n / LIFTER)
    c = mfcc / lift
    log_energy = mfcc[:, 0].copy()
    c[:, 0] = 0.0
    logmel = idct(c, type=2, axis=1, norm="ortho")
    shift = log_energy - np.log(np.exp(logmel).sum(1))
    return logmel + shift[:, None]


def logmel_to_wave_griffinlim(logmel, n_iter=200):
    """26-band log-mel energies -> waveform by least-squares band inversion and Griffin-Lim."""
    import torchaudio
    fb = torch.from_numpy(_mel_filterbank()).double()
    power = torch.linalg.lstsq(fb.T @ fb + 1e-6 * torch.eye(fb.shape[1], dtype=torch.float64), fb.T).solution
    spec = (torch.from_numpy(np.exp(logmel)).double() @ power.T).clamp_min(1e-10) * N_FFT   # (T, 257) power
    gl = torchaudio.transforms.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP, power=2.0, n_iter=n_iter,
                                          window_fn=torch.hamming_window)
    wav = gl(spec.T.float())
    out = torch.zeros_like(wav)                      # undo pre-emphasis
    acc = 0.0
    w = wav.numpy()
    o = np.zeros_like(w)
    for i in range(len(w)):
        acc = w[i] + PREEMPH * acc
        o[i] = acc
    o = o / (np.abs(o).max() + 1e-9) * 0.9
    return o


def hifigan_mel(wav):
    """Mel spectrogram in the convention of speechbrain/tts-hifigan-libritts-16kHz. wav: (B, N) in [-1, 1]."""
    import torchaudio
    tf = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0,
                                              f_max=8000.0, n_mels=80, power=1, normalized=False, norm="slaney",
                                              mel_scale="slaney").to(wav.device)
    return torch.log(torch.clamp(tf(wav), min=1e-5))             # (B, 80, L)


class Feat2Mel(nn.Module):
    """Maps a feature sequence (MFCC at 50 Hz or log-mel at 100 Hz) to the vocoder's 62.5 Hz mel."""

    def __init__(self, n_in, n_out=80, d=384, n_layers=6, n_heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(n_in, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=2))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))
        self.d = d

    def forward(self, feats, out_len, lengths=None):
        """feats: (B, T, n_in); returns (B, n_out, out_len)."""
        h = self.inp(feats.transpose(1, 2))
        h = nn.functional.interpolate(h, size=out_len, mode="linear", align_corners=False).transpose(1, 2)
        pos = torch.arange(out_len, device=h.device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d, 2, device=h.device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(out_len, self.d, device=h.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        mask = None
        if lengths is not None:
            mask = torch.arange(out_len, device=h.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.enc(h + pe, src_key_padding_mask=mask)
        return self.out(h).transpose(1, 2)


def load_hifigan(device, cache):
    from speechbrain.inference.vocoders import HIFIGAN
    return HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir=cache,
                                run_opts={"device": device})
