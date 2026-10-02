# Unsplice

[![tests](https://github.com/takakhoo/unsplice/actions/workflows/tests.yml/badge.svg)](https://github.com/takakhoo/unsplice/actions/workflows/tests.yml)

**One federated update from a speech recognizer is enough to recover what the client said. Unsplice reads the speech features out of a single gradient in closed form: {{ds1_succ}}% of LibriSpeech utterances up to 6.2 s come back at a median {{ds1_snr}} dB feature SNR in about {{ds1_sec}} s each, with no transcript, no loss function, and no optimization. Longer utterances, up to the 35 s maximum of the test set, are decoded frame by frame from deeper layers.**

Federated learning keeps audio on the device and sends only model updates. Speech models read their input through overlapping windows of frames (spliced context in DeepSpeech, a strided convolution in Transformer encoders). The first layer applies one linear map to every window, so its gradient is a sum of outer products, one per window, and its row space is the span of the windows. Because neighbouring windows share frames, "every window lies in that span" is a linear system in the unknown features. Unsplice solves it.

Audio synthesised from the recovered features is transcribed by Whisper at {{aud_ds1_rec}}% WER (the original recordings: {{aud_ds1_orig}}%), and a speaker verification model picks the right speaker out of {{aud_ds1_speakers}} for {{aud_ds1_spk}}% of utterances.

![How the attack works](docs/figures/overview.png)

**Listen:** [original](docs/audio/{{demo_id}}_original.wav) · [reconstructed from one gradient](docs/audio/{{demo_id}}_reconstructed.wav) · more in the [interactive page](https://takakhoo.github.io/unsplice/).

## The result

Every row is one update per utterance from LibriSpeech test-clean, computed in float32. The attacker gets the update and the public model weights, nothing else.

| Target | Attack | Utterances | Audio length | Recovered | Median feature SNR | Median time |
|---|---|---|---|---|---|---|
| DeepSpeech-1, random weights | closed form, first layer | {{ds1_n}} | 1.3 to 6.2 s | **{{ds1_succ}}%** | {{ds1_snr}} dB | {{ds1_sec}} s |
| DeepSpeech-1, random weights | sequential, deeper layers | {{seq_n}} | 6.3 to {{seq_max}} s | **{{seq_succ}}%** | {{seq_snr}} dB | {{seq_sec}} s |
| DeepSpeech-1, trained ({{ds1_wer}}% WER) | closed form | {{ds1t_n}} | up to 6.2 s | **{{ds1t_succ}}%** | {{ds1t_snr}} dB | |
| DeepSpeech-1, trained ({{ds1_wer}}% WER) | sequential | {{ds1ts_n}} | 6.3 to 20 s | **{{ds1ts_succ}}%** | {{ds1ts_snr}} dB | |
| DeepSpeech-1, trained, dropout 0.2 on the client | closed form | {{ds1d_n}} | up to 6.2 s | **{{ds1d_succ}}%** | {{ds1d_snr}} dB | |
| Transformer-CTC (pfl4asr front end), random weights | closed form, first conv | {{tf_n}} | up to 9.6 s | **{{tf_succ}}%** | {{tf_snr}} dB | {{tf_sec}} s |
| Transformer-CTC, trained ({{tf_wer}}% WER) | closed form | {{tft_n}} | up to 9.6 s | {{tft_succ}}% ({{tftk_succ}}% with length given) | {{tft_snr}} dB | |
| Gradient matching, the previous method | 2,000 Adam steps, transcript given | {{gm_n}} | {{gm_audio}} s | 0% | {{gm_snr}} dB | {{gm_sec}} s |

- **"Recovered" is strict:** mean absolute MFCC error below 0.01 on features whose mean magnitude is about 8 (DeepSpeech), or above 30 dB SNR on normalised log-mel (Transformer).
- **The failures have one cause.** {{ds1_fail}} of the {{ds1_n}} short utterances fail, and all {{ds1_fail_amb}} are flagged by the solver's uniqueness check: they contain digital silence (runs of bit-identical frames), which makes the solution non-unique. See [Limits](#limits).
- **The utterance length is an output.** The rank of the gradient is the number of frames. No length is given to the attacker in any row except where stated.
- **The transcript is never used.** Earlier speech attacks assume it.

**Status.** This work is being prepared for **Interspeech 2027** (São Paulo; paper deadline 9 February 2027). A 6-page version in the **ESANN 2027** format (deadline 18 November 2026) is in [`paper/esann/`](paper/esann/), and the Interspeech version is in [`paper/interspeech/`](paper/interspeech/). Neither has been submitted or peer reviewed. Interspeech does not allow the same paper to be under review elsewhere, so only one of the two will be submitted as is. An earlier manuscript on this project was not accepted at ESANN; this repository replaces its method and results (see [What happened to the earlier attack](#what-happened-to-the-earlier-attack)).

## Contents

- [How it works](#how-it-works)
- [The science, step by step](#the-science-step-by-step)
- [What a server sees in practice](#what-a-server-sees-in-practice)
- [From features to words and voices](#from-features-to-words-and-voices)
- [What stops it](#what-stops-it)
- [Other architectures](#other-architectures)
- [Against prior work](#against-prior-work)
- [Limits](#limits)
- [Try it](#try-it)
- [Reproduce](#reproduce)
- [What happened to the earlier attack](#what-happened-to-the-earlier-attack)

## How it works

A client holds an utterance with feature frames $x_1,\dots,x_T \in \mathbb{R}^F$. The model's first layer reads windows of $k$ frames taken every $s$ frames, $c_p = [x_{ps-a},\dots,x_{ps-a+k-1}]$, zero-padded at the ends, and computes $W c_p + b$ for every window with the same $W$. Whatever the rest of the network and the loss are, the chain rule gives

$$\nabla_W L = \sum_{p=1}^{P} g_p\, c_p^{\top}, \qquad \nabla_b L = \sum_{p=1}^{P} g_p, \qquad g_p = \frac{\partial L}{\partial (W c_p + b)} .$$

**1. The span.** The rows of $[\nabla_W L \mid \nabla_b L]$ are combinations of the vectors $[c_p, 1]$. When the layer is wider than the number of windows and the $g_p$ are independent, the row space equals $\mathcal{S} = \mathrm{span}\{[c_p,1]\}$, and its dimension is $P$. An SVD gives an orthonormal basis of $\mathcal{S}$ and the count $P$.

**2. The linear system.** Let $\Pi^{\perp}$ project onto the orthogonal complement of $\mathcal{S}$. The true features satisfy $\Pi^{\perp}[c_p(X), 1] = 0$ for every $p$. Each $c_p(X)$ is a linear function of $X$ (it just copies frames), so these are $P(kF+1-P)$ linear equations in the $TF$ unknowns. Windows overlap, so each frame appears in several equations and the system is banded. A banded Cholesky solve returns $X$.

**3. The capacity.** With $T \approx sP$ unknown frames, there are at least as many equations as unknowns when

$$P \;\le\; (k-s)\,F + 1 .$$

For DeepSpeech-1 with 6 context frames ($k=13$, $s=1$, $F=26$) this is 312 frames, or 6.2 s. For the Transformer front end ($k=7$, $s=3$, $F=80$) it is 320 windows, or 9.6 s. No overlap ($k = s$) means no capacity.

**4. Past the capacity.** The layer after the first sees $h(c_p)$, a nonlinear function of the same window in 2,048 dimensions (4,096 at the LSTM input), and its gradient leaks $\mathrm{span}\{[h(c_p),1]\}$ in the same way. That span holds one direction per frame up to 2,048 or 4,096 frames. The attack uses the second and third layers, whose spans are clean down to the float32 floor, and adds the LSTM input only for utterances beyond about 37 s. A candidate window can be tested against it, the zero padding identifies the first window, and each later window adds one unknown frame. Unsplice solves the first window, then extends one frame at a time (26 unknowns per step), re-fitting the last 26 frames every 8 steps, until it reads the zero padding at the far end.

## The science, step by step

### 1. The rank of the gradient is the length of the utterance

![Singular values of the first-layer gradient](docs/figures/spectrum.png)

Singular values of the first-layer gradient for two short utterances (left). The spectrum drops by four to five orders of magnitude exactly at the frame count; what is left is float32 rounding. For a 20 s utterance (right) the first layer is saturated at its 339 dimensions, but the second layer and the LSTM input still show the gap at T.

### 2. The closed form is exact up to the predicted capacity

![Error around the capacity](docs/figures/capacity_cliff.png)

Real LibriSpeech features cropped to an exact length, four front-end shapes. The relative error stays below 1e-3 up to $(k-s)F$ and jumps to order 1 within two windows of it. The prediction needs no fitted constant. The largest length that still works is {{cliff_summary}}.

### 3. Long utterances: decode one frame at a time

![True and recovered features](docs/figures/examples.png)

A 3.5 s utterance recovered in closed form and a 20 s utterance recovered by sequential decoding. The bottom row is the absolute error per frame on a log scale; the dashed line is the typical feature magnitude.

![Error and compute against length](docs/figures/duration.png)

Every test-clean utterance attacked so far. The gradient-matching baseline and the earlier paper's 10 s run sit five to six orders of magnitude above. The sequential decoder finds the first window from the opening frames of public training utterances (the attacker's own data), and it finds the length by itself: it stops when it has decoded six all-zero frames, which is the padding.

### 4. Training does not close the leak

![Attack success along training](docs/figures/training.png)

The same attack on checkpoints saved while training each model on LibriSpeech (100 h, then 460 h). The closed form depends on the weights only through the backward signals $g_p$, which need to be independent, not informative.

{{ckpt_table}}

For the Transformer the picture is less clean: a float32 convolution backward pass leaves one stray direction in the gradient that sits close to the weakest real direction on trained weights, so the attacker sometimes miscounts the windows by one. At the last 100 h checkpoint {{tft_succ}}% of utterances are recovered without the length and {{tftk_succ}}% with it.

## What a server sees in practice

![Closed form under realistic updates](docs/figures/fl_surfaces.png)

A server rarely receives a clean single gradient. It receives weight differences after local training, over batches.

- **Local SGD steps do not help the client.** The first-layer update after K steps is $\sum_k \sum_p g_p^{(k)} c_p^{\top}$: the windows do not depend on the weights, so the row space is the same span. Momentum and weight decay keep it (the server knows the learning rate and undoes the decay).
- **Batches share one budget.** Windows of all utterances in the update land in one span. The closed form works while the total number of frames stays within capacity. In these runs the attacker is given the lengths of the utterances in the batch; the sequential decoder does not need them.
- **Clipping does nothing.** It rescales the update; the span is unchanged.
- **The floor is arithmetic.** A server that subtracts float32 weights loses precision in proportion to how small the step was. With a learning rate of 0.01 the error is around 5e-3; with 0.1 it is around 3e-4.
- **Local Adam breaks it.** Adam rescales every coordinate separately, which destroys the low-rank structure.

## From features to words and voices

![Words and speaker from the reconstructed audio](docs/figures/audio.png)

The recovered features are turned into audio by a small network that maps them to the mel spectrogram of a public HiFi-GAN vocoder (trained by the attacker on LibriSpeech train-clean-100, which shares no speakers with the test set). Whisper `small.en` then transcribes the audio, and ECAPA-TDNN embeddings assign each reconstruction to one of the {{aud_ds1_speakers}} test-clean speakers, enrolled from recordings that were not attacked.

{{audio_table}}

Because the features are recovered exactly, audio from recovered features and audio from the true features give the same numbers. Whatever a speech front end keeps, the gradient gives away.

## What stops it

![Noise against the weakest singular value](docs/figures/noise.png)

- **Noise, at a predictable level.** Additive Gaussian noise hides the span once its spectral norm, $\sigma(\sqrt{m}+\sqrt{n})$ for an $m \times n$ gradient, reaches the weakest signal singular value $s_T$. Below that the error grows linearly with the noise. For utterances of a few seconds this happens at a noise standard deviation near 1e-3 of the gradient's RMS value. Central differential privacy adds its noise after aggregation, so it does not protect a client from the server itself.
- **Magnitude pruning** of half or more of the entries, and **sign compression**, break the low-rank structure and the closed form fails ({{prune_succ}}% and {{sign_succ}}% recovered). We did not try low-rank completion against pruning.
- **Coarse arithmetic.** 8-bit quantisation of the update stops the attack. Half precision and 16-bit quantisation do not hide the speech but cost exactness: the median error rises to {{f16_mae}} and {{q16_mae}} (features have magnitude about 8), with the length given.
- **Aggregation past the capacity.** A sum over clients helps only when the total audio in it exceeds what the widest leaking layer can hold.
- **Narrow or non-overlapping front ends.** See the next section.
- **Dropout does not stop the closed form.** With dropout {{drop_rates}} active on the client, {{drop_succ}}% of utterances are still recovered. Dropout acts after the first layer, and the first-layer gradient still has rows in the span of the windows. Dropout was reported to reduce speaker identification from DeepSpeech gradients to 0% [Dang et al., 2022]; that result holds for optimization-based matching and does not carry over. Dropout does stop our sequential decoder ({{dropseq_succ}}% of {{dropseq_n}} long utterances recovered): the deeper layers see the dropped activations, so their spans no longer contain the clean lifted frames.

## Other architectures

Whether a model leaks this way is decided by the shapes of its first layers. For one real 5 s utterance, the rank of early-layer gradients in several ASR front ends (Whisper and wav2vec 2.0 with their released weights):

{{probe_table}}

A layer identifies its inputs only when its gradient has rank equal to the number of windows and below both of its dimensions. Whisper pads every input to 3,000 frames, wav2vec 2.0 reads 10-sample windows of raw audio, DeepSpeech-2 has 32 first-layer channels, and QuartzNet's first layer is depthwise; none of these first layers leaks a usable span. Several of them leak one layer later: the wav2vec 2.0 feature projection and the Conformer projection after subsampling have gradients whose rank is the frame count.

**Conformer-style front end.** For Conv2dSubsampling followed by a linear projection (ESPnet, WeNet and NeMo Conformers), the projection's gradient leaks the span of 7-frame receptive fields. A decoder that finds one window from public speech and extends it four frames at a time does find true windows (the seed lands on a real window at the first attempt), but the chains stall: on {{conf_n}} utterances of {{conf_len}} (random weights, d = {{conf_d}}) a single chain recovers {{conf_cov}}% of the frames on average, at {{conf_snr}} dB feature SNR for the part it recovers. Seeding several chains and merging them reached 92% of one 6 s utterance, after 41 minutes. We count this as a mostly negative result: the leak is there, and our decoder does not yet exploit it.

## Against prior work

| | Target | Loss | Needs transcript | Needs length | Max length shown | Method | Reported quality |
|---|---|---|---|---|---|---|---|
| Dang et al., ICASSP 2022 | DeepSpeech | CTC | yes | yes | short utterances | zeroth-order gradient matching | speaker ID 34% top-1 |
| Li et al., ICASSP 2023 | keyword-spotting CNN | cross-entropy | label | fixed 1 s | 1 s | gradient matching | MFCC: PESQ 1.39 |
| Zeng and Rudzicz, Interspeech 2025 | 4-layer CNN classifier | cross-entropy | dummy label | segments | 139 s | segment-wise gradient matching | 40 dB SNR on mel |
| Earlier manuscript of this project | DeepSpeech-1 without ReLU | CTC | yes | yes | 10 s | grid gradient matching | MFCC MAE 8.78, 3,200 s |
| **Unsplice** | DeepSpeech-1 with ReLU; Transformer-CTC | any | **no** | **no** | **35 s** | closed form; span decoding | MFCC MAE {{seq_mae}}, speaker ID {{aud_ds1_spk}}% |

Numbers for other papers are the ones they report, on their own data. The low-rank structure itself is known from DAGER and SPEAR (exact recovery of text tokens and image batches) and TIGER (subspace-distance optimization for language models). What is new here is the time axis: frames of one utterance play the role of a batch, the overlap of context windows turns the span into a linear system with a closed-form capacity, and the zero padding plus overlap give an ordering that lets long utterances be decoded sequentially.

## Limits

- **Digital silence.** Runs of 13 or more bit-identical frames produce identical windows. The rank then undercounts the length and the linear system has more than one solution (for example, stretches of speech on either side of silence can trade places). The solver detects this with a uniqueness check and the result is counted as a failure. This affects {{silence_pct}}% of test-clean, all in recordings with edited-in silence.
- **Capacity.** The closed form stops at $(k-s)F$ windows in total across everything in the update.
- **Features, not waveforms.** The attack recovers what the front end computes. 26 MFCCs do not carry pitch; the vocoder infers it.
- **Precision of the update.** Results are for float32 gradients. Updates sent after tiny local steps lose precision to rounding.
- **Trained Transformer.** Without the length, recovery at the trained checkpoint is {{tft_succ}}%.
- **Not attacked.** Whisper, wav2vec 2.0, DeepSpeech-2 and QuartzNet first layers do not leak a usable span, and no attack on them is claimed. The Conformer decoder mostly fails.
- **Long utterances under dropout** are not recovered.
- **One client per update.** Secure aggregation over enough clients exceeds every capacity measured here.

## Try it

```python
import torch
from unsplice.model import DeepSpeech1, client_gradient, text_to_ids
from unsplice.linear_attack import ds1_attack

model = DeepSpeech1()                               # the public model
x = torch.randn(200, 26)                            # a client's 4 s of MFCCs
update, _ = client_gradient(model, x.unsqueeze(1), text_to_ids("the client transcript"))

X, info = ds1_attack(update)                        # the attacker sees only `update`
print(info["T"], float((X - x).abs().max()))        # 200, about 1e-6
```

## Reproduce

```bash
pip install -e . && pytest tests -q                  # 8 CPU checks of the core claims, about 6 s
python experiments/prep_features.py $LIBRI test-clean feats
python experiments/run_ds1.py --feats feats --out results/raw/e1/run.jsonl --max-frames 312     # closed form
python experiments/run_ds1.py --feats feats --out results/raw/e2/run.jsonl --min-frames 313     # sequential
python experiments/run_tf.py  --feats feats --out results/raw/tfatk/run.jsonl --max-frames 960  # Transformer
python experiments/make_summary.py && python experiments/make_figures.py && python experiments/render_readme.py
```

Every number in this README comes from [`results/summary.json`](results/summary.json), which `make_summary.py` builds from the per-utterance tables in [`results/raw/`](results/raw/). [`results/README.md`](results/README.md) says what each file is.

## What happened to the earlier attack

The earlier manuscript matched the output-layer gradient by gradient descent through a twice-differentiable CTC loss, on a DeepSpeech-1 whose fully connected layers had no nonlinearity, with the transcript and length given. Its best recorded 10 s run had MFCC MAE 8.78 and a negative feature SNR after 3,200 s. Rerun here on 1.8 s utterances, the same procedure ends at MAE {{gm_mae}} after {{gm_sec}} s per utterance. The gradient fit is good and the features are wrong: the output layer's gradient has rank at most 29, far too little to pin down hundreds of frames. The twice-differentiable CTC is kept in `unsplice/ctc2.py` for that baseline only.

## Responsible use

This is privacy research on public data (LibriSpeech). The point is to measure what federated speech training exposes so that systems can be designed against it: avoid wide linear layers on overlapping feature windows at the input, use secure aggregation with enough clients, and add noise before the server sees an update. Use the code only on models and data you are authorised to test.

## Credits

LibriSpeech (Panayotov et al.). DeepSpeech (Hannun et al.) and the Myrtle PyTorch port used by the earlier code. The Transformer front end follows Apple's pfl4asr (Pelikan et al.). HiFi-GAN vocoder and ECAPA-TDNN from SpeechBrain; Whisper from OpenAI. The low-rank view of gradients follows DAGER, SPEAR and TIGER (Petrov, Dimitrov, Kalikman, Vechev and co-authors). Full references are in [`paper/refs.bib`](paper/refs.bib) and the survey in [`docs/literature.md`](docs/literature.md).
