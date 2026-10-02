# Results

Everything here is produced by scripts in `experiments/`. `summary.json` is the authority for every number in the README, the page and the papers.

| Path | What it is | Produced by |
|---|---|---|
| `summary.json` | All aggregate numbers | `experiments/make_summary.py` |
| `raw/e1/*.jsonl` | Closed-form attack on DeepSpeech-1, one line per test-clean utterance up to 312 frames | `experiments/run_ds1.py --max-frames 312` |
| `raw/e2b/*.jsonl` | Sequential decoding on DeepSpeech-1, utterances over 312 frames (including the six longest), with the default fc2 and fc3 spans | `experiments/run_ds1.py --min-frames 313` |
| `raw/e2/*.jsonl` | The first run of the same sweep, which also used the LSTM-input span (about 50x less precise; kept for the record) | earlier revision of `unsplice/seq_attack.py` |
| `raw/tfatk/*.jsonl` | Closed-form attack on the Transformer-CTC front end | `experiments/run_tf.py` |
| `raw/ckpt/*.jsonl` | The same attacks on checkpoints saved during training (`ds1_*`, `tf_*`; `tfknown_*` gives the attacker the length; `*460*` continues training on 460 h) | `run_ds1.py --ckpt`, `run_tf.py --ckpt` |
| `raw/ds1/`, `raw/tf/`, `raw/ds1_460/`, `raw/tf_460/` | Training logs with dev-clean WER | `experiments/train_asr.py` |
| `raw/fl/*.jsonl` | Local SGD steps, optimizers, batches, over-capacity batches | `experiments/fl_surfaces.py` |
| `raw/def/*.jsonl` | Noise, pruning, sign, quantisation, clipping, dropout. `*_known` gives the attacker the length, `*_blind` does not | `run_ds1.py --defense`, `--train-mode --dropout` |
| `raw/def/noise_law_*.jsonl` | Error against noise level relative to the weakest signal singular value | `experiments/noise_law.py` |
| `raw/capacity/cliff.jsonl` | Error around the predicted capacity for four front-end shapes | `experiments/capacity_cliff.py` |
| `raw/capacity/probe.json` | Gradient ranks of early layers in seven architectures | `experiments/capacity_probe.py` |
| `raw/baseline/*.jsonl` | Gradient matching through a twice-differentiable CTC, with loss curves | `experiments/baseline_gm.py` |
| `raw/conformer/*.jsonl` | Span decoding on a Conformer-style front end | `experiments/run_conformer.py` |
| `raw/audio/*/` | Whisper WER and speaker identification of synthesised reconstructions, per utterance and in summary | `experiments/eval_audio.py` |

## Reading a per-utterance line

`T` is the true number of frames and `T_est` the attacker's. `mae`, `rel` and `snr_db` compare recovered and true features. `length_error` is `T_est - T`; when it is negative the reconstruction was aligned to the truth before scoring (see `unsplice/metrics.py`). `ambiguity` is the solver's uniqueness probe: values above 1e-2 mean the linear system has more than one solution. `seconds` is wall time on a GPU shared with other jobs.

## Protocol

- One float32 gradient of the CTC loss per utterance, computed on the client model. The attack functions receive the update and the model, never the features or transcript.
- Public data for the attacker (starting points for the sequential decoder, vocoder front end): LibriSpeech train-clean-100, which shares no speakers with test-clean.
- "Recovered" means MAE below 0.01 for MFCC features and SNR above 30 dB for normalised log-mel.
- Utterance order is a fixed random permutation (seed 1234), so `--limit` takes a random sample.

## Corrections

- **Batch scenarios.** The first run of the batch-of-4 and batch-of-8 scenarios cropped every utterance to the same length. Equal lengths make any reordering of the utterances a solution, and the solver returned a blend. The scenarios were rerun with distinct lengths (`fl/random0_batch.jsonl`); the stale rows in `fl/random0.jsonl` are not used.
- **Transformer checkpoints.** The first run of the trained-Transformer attacks counted a float32 convolution artefact as a window and misjudged the length on about half the utterances. The rank selection was fixed (`conv1d_attack` fits both readings of the spectrum) and the runs repeated.
- **Speaker identification.** A preliminary evaluation enrolled speakers from the attacked utterances themselves (22 speakers, 39 utterances, 90% top-1). The reported protocol enrols all 40 test-clean speakers from recordings that were not attacked.
