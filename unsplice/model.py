"""DeepSpeech-1 acoustic model, written out explicitly so every layer's input is easy to name.

Layout follows Hannun et al. (2014) and the Myrtle PyTorch port used by the earlier
gradient-matching attack: three per-frame fully connected layers on spliced MFCCs,
one bidirectional LSTM, one more per-frame layer, and a linear output layer.
"""
import torch
import torch.nn as nn

ALPHABET = "_ abcdefghijklmnopqrstuvwxyz'"  # index 0 is the CTC blank
BLANK = 0


def text_to_ids(text):
    return torch.tensor([ALPHABET.index(ch) for ch in text.lower() if ch in ALPHABET[1:]], dtype=torch.long)


def ids_to_text(ids):
    return "".join(ALPHABET[i] for i in ids)


def splice(x, n_context):
    """Stack each frame with n_context neighbours on both sides, zero-padded at the ends.

    x: (T, B, F) -> (T, B, (2*n_context+1)*F). Block k of row t holds frame t + k - n_context.
    """
    T, B, F = x.shape
    pad = x.new_zeros(n_context, B, F)
    xp = torch.cat([pad, x, pad], dim=0)
    return torch.cat([xp[k:k + T] for k in range(2 * n_context + 1)], dim=-1)


class DeepSpeech1(nn.Module):
    def __init__(self, n_feat=26, n_context=6, n_hidden=2048, n_out=len(ALPHABET),
                 relu=True, relu_clip=20.0, forget_gate_bias=1.0, dropout=0.0):
        super().__init__()
        self.n_feat, self.n_context, self.n_hidden = n_feat, n_context, n_hidden
        self.relu, self.relu_clip = relu, relu_clip
        self.drop = nn.Dropout(dropout)
        d = n_feat * (2 * n_context + 1)
        self.fc1 = nn.Linear(d, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 2 * n_hidden)
        self.lstm = nn.LSTM(2 * n_hidden, n_hidden, bidirectional=True)
        self.fc4 = nn.Linear(2 * n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        if forget_gate_bias is not None:
            for name in ("bias_ih_l0", "bias_ih_l0_reverse"):
                getattr(self.lstm, name).data[n_hidden:2 * n_hidden].fill_(forget_gate_bias)
            for name in ("bias_hh_l0", "bias_hh_l0_reverse"):
                getattr(self.lstm, name).data[n_hidden:2 * n_hidden].fill_(0)

    def act(self, a):
        return torch.clamp(a, 0.0, self.relu_clip) if self.relu else a

    def front(self, x, upto=3):
        """Per-frame stack before the LSTM. Returns [c, h1, h2, h3][: upto + 1]."""
        feats = [splice(x, self.n_context)]
        for fc in (self.fc1, self.fc2, self.fc3)[:upto]:
            feats.append(self.drop(self.act(fc(feats[-1]))))
        return feats

    def forward(self, x, return_hidden=False):
        h3 = self.front(x)[-1]
        r, _ = self.lstm(h3)
        h4 = self.drop(self.act(self.fc4(r)))
        z = self.out(h4)
        return (z, h4) if return_hidden else z


def client_gradient(model, x, targets, loss="ctc"):
    """What a federated client sends for one utterance: d loss / d theta for every parameter.

    x: (T, 1, F) features, targets: 1-D LongTensor of label ids.
    """
    model.zero_grad(set_to_none=True)
    z = model(x)
    T = z.shape[0]
    lp = z.log_softmax(-1)
    val = nn.functional.ctc_loss(lp, targets.unsqueeze(0), torch.tensor([T]), torch.tensor([len(targets)]),
                                 blank=BLANK, reduction="mean", zero_infinity=False)
    grads = torch.autograd.grad(val, list(model.parameters()))
    return {n: g.detach() for (n, _), g in zip(model.named_parameters(), grads)}, float(val.detach())
