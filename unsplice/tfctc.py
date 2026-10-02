"""Transformer-CTC encoder with the front end of Apple's pfl4asr benchmark model.

Layout: per-utterance normalised 80-bin log-mel -> Conv1d(80, 2d, kernel 7, stride 3) -> GLU
-> sinusoidal positions -> pre-LN Transformer blocks -> LayerNorm -> Linear to characters.
The first convolution is a per-patch affine map on 7 stacked frames, which is the layer the
closed-form attack reads.
"""
import math
import torch
import torch.nn as nn

KERNEL, STRIDE, PAD = 7, 3, 3


def normalise(x, lengths=None):
    """Per-utterance mean/variance normalisation over frames and channels. x: (B, T, F)."""
    if lengths is None:
        m, s = x.mean(dim=(1, 2), keepdim=True), x.std(dim=(1, 2), keepdim=True)
        return (x - m) / s
    out = torch.zeros_like(x)
    for i, n in enumerate(lengths):
        seg = x[i, :n]
        out[i, :n] = (seg - seg.mean()) / seg.std()
    return out


class TransformerCTC(nn.Module):
    def __init__(self, n_mels=80, d_model=768, n_layers=12, n_heads=4, d_ff=3072, n_out=29, dropout=0.1):
        super().__init__()
        self.n_mels, self.d_model = n_mels, d_model
        self.conv = nn.Conv1d(n_mels, 2 * d_model, KERNEL, stride=STRIDE, padding=PAD)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln_final = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_out)

    @staticmethod
    def out_lengths(lengths):
        return (lengths + 2 * PAD - KERNEL) // STRIDE + 1

    def positions(self, P, device, dtype):
        pos = torch.arange(P, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(P, self.d_model, device=device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.to(dtype)

    def embed(self, x):
        """x: (B, T, F) normalised features -> (B, P, d) block-0 inputs, before dropout."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        a, b = h.chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        return h + self.positions(h.shape[1], h.device, h.dtype)

    def forward(self, x, lengths=None):
        h = self.drop(self.embed(x))
        mask = None
        if lengths is not None:
            P = self.out_lengths(lengths)
            mask = torch.arange(h.shape[1], device=h.device).unsqueeze(0) >= P.unsqueeze(1)
        h = self.blocks(h, src_key_padding_mask=mask)
        return self.linear(self.ln_final(h))
