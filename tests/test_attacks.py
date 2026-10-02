"""CPU checks of the claims the project rests on. Small models, synthetic features, a few seconds."""
import copy
import torch
import torch.nn as nn

from unsplice.model import DeepSpeech1, client_gradient
from unsplice.linear_attack import ds1_attack, conv1d_attack, estimate_rank, gradient_rowspace, solve_patches
from unsplice.seq_attack import SpanDecoder
from unsplice.defenses import apply_defense

torch.manual_seed(0)


def small_ds1(**kw):
    return DeepSpeech1(n_feat=8, n_context=2, n_hidden=160, **kw)     # windows of 5 frames, 40 dims, capacity 32


def utterance(T, F=8):
    x = torch.randn(T, F) * 3
    y = torch.randint(1, 29, (max(2, T // 4),))
    return x, y


def test_closed_form_recovers_features_and_length_without_labels():
    model = small_ds1()
    x, y = utterance(24)
    grads, _ = client_gradient(model, x.unsqueeze(1), y)
    X, info = ds1_attack(grads, n_feat=8, n_context=2)       # sees only the update
    assert info["T"] == 24
    assert (X.float() - x).abs().max() < 1e-2


def test_rank_of_first_layer_gradient_is_the_frame_count():
    model = small_ds1()
    for T in (7, 15, 30):
        x, y = utterance(T)
        grads, _ = client_gradient(model, x.unsqueeze(1), y)
        s, _ = gradient_rowspace(grads["fc1.weight"], grads["fc1.bias"])
        assert estimate_rank(s) == T


def test_capacity_is_kernel_minus_stride_times_features():
    model = small_ds1()
    cap = (5 - 1) * 8
    errs = {}
    for T in (cap, cap + 3):
        x, y = utterance(T)
        grads, _ = client_gradient(model, x.unsqueeze(1), y)
        X, _ = ds1_attack(grads, n_feat=8, n_context=2, T=T)
        errs[T] = float((X.float() - x).norm() / x.norm())
    assert errs[cap] < 1e-2 and errs[cap + 3] > 0.1


def test_local_sgd_steps_leave_the_span_unchanged():
    model = small_ds1()
    x, y = utterance(20)
    work = copy.deepcopy(model)
    opt = torch.optim.SGD(work.parameters(), lr=0.05)
    for _ in range(5):
        opt.zero_grad()
        z = work(x.unsqueeze(1))
        nn.functional.ctc_loss(z.log_softmax(-1), y.unsqueeze(0), torch.tensor([20]), torch.tensor([len(y)])).backward()
        opt.step()
    before = dict(model.named_parameters())
    update = {n: (before[n] - p).detach() / 0.05 for n, p in work.named_parameters()}
    X, _ = ds1_attack(update, n_feat=8, n_context=2, T=20)
    assert (X.float() - x).abs().mean() < 0.05


def test_dropout_and_clipping_do_not_protect_the_first_layer():
    model = small_ds1(dropout=0.3)
    model.train()
    x, y = utterance(22)
    grads, _ = client_gradient(model, x.unsqueeze(1), y)
    grads = apply_defense(grads, "clip:0.001")
    X, info = ds1_attack(grads, n_feat=8, n_context=2)
    assert info["T"] == 22 and (X.float() - x).abs().max() < 1e-2


def test_strided_conv_front_end():
    F, k, s, pad, C = 12, 7, 3, 3, 256
    conv = nn.Conv1d(F, C, k, stride=s, padding=pad)
    head = nn.Sequential(nn.ReLU(), nn.Linear(C, 64), nn.ReLU(), nn.Linear(64, 29))
    T = 91
    x = torch.randn(T, F)
    z = head(conv(x.T.unsqueeze(0)).squeeze(0).T)
    y = torch.randint(1, 29, (6,))
    loss = nn.functional.ctc_loss(z.log_softmax(-1).unsqueeze(1), y.unsqueeze(0), torch.tensor([z.shape[0]]), torch.tensor([6]))
    gW, gb = torch.autograd.grad(loss, [conv.weight, conv.bias])
    X, info = conv1d_attack(gW, gb, F, k, s, pad)
    assert info["T"] == T
    assert (X.float() - x).abs().max() < 1e-2


def test_sequential_decoding_past_the_first_layer_limit():
    model = small_ds1()
    T = 60                                         # capacity of the first layer is 32
    x, y = utterance(T)
    grads, _ = client_gradient(model, x.unsqueeze(1), y)
    dec = SpanDecoder(model, grads)
    inits = torch.randn(12, 3, 8) * 3
    X, info = dec.decode(inits=inits, last_inits=inits)
    assert info["T"] == T
    X, _ = dec.polish(X)
    assert (X.float() - x).abs().mean() < 1e-2


def test_uniqueness_check_flags_digital_silence():
    """Two runs of identical frames can trade length with each other, so the solution is not unique."""
    model = small_ds1()
    x, y = utterance(30)
    x[4:11] = x[4]
    x[16:23] = x[4]
    grads, _ = client_gradient(model, x.unsqueeze(1), y)
    s, vh = gradient_rowspace(grads["fc1.weight"], grads["fc1.bias"])
    rank = estimate_rank(s)
    assert rank < 30                                 # identical windows: the rank undercounts the length
    _, amb = solve_patches(vh[:rank], 8, 30, 5, 1, 2, check_unique=True)
    x2, _ = utterance(30)
    g2, _ = client_gradient(model, x2.unsqueeze(1), y)
    s2, vh2 = gradient_rowspace(g2["fc1.weight"], g2["fc1.bias"])
    _, amb2 = solve_patches(vh2[:30], 8, 30, 5, 1, 2, check_unique=True)
    assert amb > 1e-2 > amb2
