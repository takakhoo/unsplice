# What was tried, in order

A record of the approaches behind the results, including the ones that failed. Numbers are from the runs in `results/raw/`.

## 1. The starting point

The earlier attack optimised a dummy MFCC sequence until the output-layer gradient matched, through a twice-differentiable CTC loss, with the transcript given and the DeepSpeech-1 fully connected layers stripped of their nonlinearity. Its best recorded 10 s run: MFCC MAE 8.78, negative feature SNR, 3,200 s. Rerun here (`experiments/baseline_gm.py`) on 1.8 to 2.3 s utterances it ends at MAE 5.9 to 8.0 after about 50 minutes each, with a cosine distance below 0.01. The gradient is matched and the input is not: the output layer's gradient has rank at most 29.

Matching all layers ran out of memory (more than 34 GB at 100 frames) because the LSTM has to be differentiated twice without cuDNN.

## 2. The first layer is linear in the windows

The first-layer gradient is a sum of outer products, one per context window. Its row space is the span of the windows, and the windows overlap. Writing "every window of X lies in the span" as equations in X gives a banded linear system. First test on random features: exact up to 312 frames, wrong at 313. The limit is (k − s)F, which the same code then confirmed on three other front-end shapes.

Things that went wrong on the way:

- **Rank from a weight difference.** A server that subtracts float32 weights sees a noise floor near 4e-6 of the top singular value for a learning rate of 0.01. The first rank rule needed a 30x gap and missed it. The rule now takes the largest drop.
- **A stray direction in convolution gradients.** A float32 Conv1d backward pass leaves one direction between signal and noise that vanishes in float64. On random weights it is 50x below the signal; on trained weights only 3x. The Transformer attack now fits both readings of the spectrum and keeps the smaller rank when it fits as well.
- **Digital silence.** Two LibriSpeech speakers have recordings with runs of bit-identical frames. Identical windows lower the rank, and the linear system then has several solutions. There is no fix for this in the first layer; the solver detects it by nudging the solve and checking whether the answer moves.
- **Equal-length batches.** A batch of utterances with the same length has a permutation symmetry, and a linear solver returns a blend of the orderings. The batch experiments use distinct lengths.

## 3. Past the first-layer limit

**Joint descent on all frames (failed).** Minimising the distance of every frame's lifted features to the spans of fc2, fc3 and the LSTM input. Version one divided by a detached norm and the optimiser killed every ReLU. Version two (scale-invariant) drove the residual to 1e-2 and the features stayed wrong: every frame can sit anywhere in the span, so all frames drift to the same typical vector. The objective has no notion of which frame is which.

**Sequential decoding (works).** The zero padding identifies the first window, and each later window adds one frame. Levenberg-Marquardt on 26 unknowns per step.

- The first window (182 unknowns) failed from random starts once utterances passed about 15 s. Starting from the opening frames of public training utterances fixed it; most utterances need one try.
- The acceptance threshold was below the float32 residual floor at first, so every step used all its restarts. Fixing that halved the time.
- The decoder originally took the length from the rank. It now runs until it decodes six zero frames, the padding at the other end.
- On trained weights the chain drifted and stalled about 40 frames in. Re-fitting the last 26 frames jointly every 8 steps fixed it.
- On the most-trained checkpoint the chain still stalled after 50 to 80 frames. The cause was the LSTM-input span: its weakest real direction sits at the float32 noise level on trained weights, and its residual injected error at every step. With the fc2 and fc3 spans only, the same checkpoint decodes completely, and the error on random weights drops from 9e-5 to about 2e-6. The LSTM span is now used only for utterances over 1,850 frames, which the other two layers cannot hold. The first full sweep (`results/raw/e2/`) used all three spans; the reported one (`e2b/`) uses the default.
- Dropout on the client breaks it: the deeper spans contain dropped activations, which the attacker cannot reproduce.

## 4. Other architectures

- **Transformer-CTC with a strided Conv1d (works).** Same closed form with k = 7, s = 3, F = 80.
- **Conformer-style Conv2dSubsampling + Linear (mostly fails).** The projection leaks the span of 7-frame receptive fields and a seed window is found at the first attempt, but chains of four new frames per step (320 unknowns) stall after a few steps. Merging several seeded chains covered 92% of one utterance in 41 minutes.
- **Whisper, wav2vec 2.0, DeepSpeech-2, QuartzNet (no attack).** Their first layers are saturated: too few channels, too small a window, or no overlap. Measured, not attacked.

## 5. Checks on our own results

- The attack functions take the update and the model. Features and transcripts are used only by the client step and the scoring.
- The attacker's public data is train-clean-100, which shares no speakers with test-clean.
- Speaker identification was first run with speakers enrolled from the attacked utterances (90% over 22 speakers). The reported protocol enrols all 40 speakers from separate recordings.
- Audio from recovered features and from true features scores the same, as it must when recovery is exact.
- All 8 unit tests run on CPU in about 6 s and cover exact recovery, the rank-length property, the capacity, local SGD, dropout and clipping, the conv front end, sequential decoding, and the uniqueness flag.
