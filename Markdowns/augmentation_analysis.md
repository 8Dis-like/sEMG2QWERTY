# Analysis of Performance Degradation with "Plus" Data Augmentation

## Observation

Both the GRU and LSTM CTC models perform worse when trained with `log_spectrogram_plus` compared to the baseline `log_spectrogram` pipeline (from `results/legacy/experiments.csv` and `best_model.txt`):

| Model         | `log_spectrogram` CER | `log_spectrogram_plus` CER | Delta   |
|---------------|----------------------|---------------------------|---------|
| CNN/RNN-LSTM  | 13.33%               | 14.37%                    | +1.04%  |
| CNN/RNN-GRU   | 15.77%               | 17.33%                    | +1.56%  |

## Augmentation Pipeline Comparison

The key structural difference is **where** each augmentation acts relative to the spectrogram computation.

**Baseline (`log_spectrogram`)**:
1. `ToTensor`
2. `RandomBandRotation` — spatial shift by 1 channel
3. `TemporalAlignmentJitter` — shifts left/right electrode bands by up to 60ms
4. `LogSpectrogram` ← *domain boundary*
5. `SpecAugment` — time and frequency masking on the spectrogram

**Plus additions (`log_spectrogram_plus`)** — all injected in the **raw signal domain** before step 4:
- `RandomGain` — per-channel and per-band amplitude scaling
- `AdditiveGaussianNoise` — broadband additive noise
- `RandomBurstNoise` — short bursts of high-variance noise
- `ChannelDropout` — zeros out entire electrode channels
- `TemporalChannelDropout` — zeros out specific channels for brief windows
- `RandomTimeWarp` — stretches/compresses the signal in time
- `RandomFrequencyEQ` — applies a smooth frequency response distortion

The full "plus" pipeline is therefore the baseline with these seven additional transforms inserted before `LogSpectrogram`.

## Hypotheses for Performance Degradation

### 1. Destruction of Discriminative Spatial Information

`ChannelDropout` and `TemporalChannelDropout` completely zero out electrode channels. In forearm sEMG for typing, individual channels have strong spatial correspondence to specific muscle groups (e.g., flexor digitorum superficialis, extensor digitorum). Adjacent keystrokes (e.g., 'U' vs. 'I', 'H' vs. 'J') are distinguished by subtle differences in which channels activate. Blanking a channel removes exactly this discriminative information, making the classification task locally ambiguous. The model must learn to be robust to this loss, but the evaluation data has all channels intact — so this "robustness" hurts at test time.

### 2. Temporal Distortion Degrades CTC Alignment

`RandomTimeWarp` stretches and compresses the raw signal before spectrogram computation. CTC decoding implicitly assumes the model's output frame rate corresponds meaningfully to keystroke timing. Keystrokes are brief, discrete muscle activations; warping their duration distorts feature boundaries and can break the temporal ordering assumptions the CTC loss relies on. This is distinct from speech, where phonemes are longer and more tolerant of stretching. sEMG typing events are too brief for aggressive time warping to be safe.

### 3. Train/Test Distribution Mismatch

`AdditiveGaussianNoise`, `RandomBurstNoise`, and `RandomFrequencyEQ` shift the training distribution toward noisier, spectrally distorted signals. If the evaluation set reflects cleaner, real-world recording conditions, the model learns noise-invariance that is not rewarded at test time. The model's representational capacity is spent on sources of variation that are absent in the data it is ultimately scored on.

### 4. Over-Regularization Given Model Capacity (Speculative)

Seven additional aggressive augmentations substantially increase effective training difficulty. Models with moderate capacity trained under very heavy augmentation often underfit the core task. If training duration and learning rate schedules were not retuned from the baseline, the "plus" models may simply be under-converged relative to the difficulty of the expanded augmentation space. This hypothesis would require ablation experiments (e.g., longer training on "plus") to confirm.

## Recommendations

- **Ablate augmentations individually** — the combined pipeline makes it impossible to isolate which transform is most harmful. Running each augmentation in isolation against the baseline would identify the primary culprits.
- **Disable `ChannelDropout` variants first** — these are the most likely to destroy information required for test-time performance given the train/test mismatch argument.
- **Validate time warp range** — if `RandomTimeWarp` is retained, its stretch factor should be calibrated to the typical duration of keystroke activations, not borrowed from speech augmentation defaults.
- **Match training distribution to evaluation conditions** — if the evaluation set is clean, noise augmentations (`AdditiveGaussianNoise`, `RandomBurstNoise`) should be applied conservatively or dropped.
- **Retune training schedule before concluding "plus" is harmful** — hypothesis 4 means the current results may underestimate the potential benefit of the richer augmentation set under a properly scaled training budget.
