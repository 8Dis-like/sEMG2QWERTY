# Design: `log_spectrogram_calibrated` Augmentation Pipeline

## Motivation

`log_spectrogram_plus` degrades performance relative to the baseline on all tested architectures:

| Model        | Baseline CER | Plus CER | Delta   |
|--------------|-------------|----------|---------|
| CNN/RNN-LSTM | 13.33%      | 14.37%   | +1.04%  |
| CNN/RNN-GRU  | 15.77%      | 17.33%   | +1.56%  |

The analysis in `augmentation_analysis.md` identified four root causes:
1. **Spatial information destruction** — `ChannelDropout`/`TemporalChannelDropout` zero entire channels, removing exactly the discriminative spatial features the model relies on.
2. **CTC alignment disruption** — `RandomTimeWarp` distorts brief keystroke bursts, whose precise timing is critical for CTC decoding.
3. **Train/eval distribution mismatch** — `RandomBurstNoise` adds synthetic artifacts that do not appear in clean evaluation recordings.
4. **Over-regularization** — The cumulative effect of seven additional augmentations causes underfitting in models with moderate capacity.

`log_spectrogram_calibrated` replaces or removes every problematic augmentation, substituting physically-motivated alternatives that target **real** sources of intra-class variability while preserving all discriminative information.

---

## Design Principles

1. **Preserve spatial discriminability**: Augmentations must never destroy relative channel information. Keystrokes 'U' vs. 'I' differ by which spatial channels activate — those differences must remain detectable after augmentation.
2. **Match the evaluation distribution**: Only augment with noise/distortion that plausibly exists in the recording conditions used for evaluation. Synthetic artifacts (burst noise, heavy time warp) do not.
3. **Target real physical variability**: Good augmentations should correspond to actual sources of sEMG session-to-session variation — electrode contact quality, amplitude drift, placement shift.
4. **Conservative is better than aggressive**: With models of moderate capacity (384 hidden units, 2–4 layers), underfitting from over-augmentation is a concrete risk. Each augmentation must earn its place.

---

## Pipeline Design

### Full Train Pipeline

```
ToTensor
  ↓ (T, 2, 16) raw EMG
RandomGain          [conservative: ×0.9–1.1 per-channel/band]
  ↓
AmplitudeEnvelope   [NEW: slow temporal drift, p=0.5, ±30%]
  ↓
SoftChannelAttenuation  [NEW: attenuate p=10% of channels to 0.3–0.8×]
  ↓
AdditiveGaussianNoise   [conservative: σ ≤ 0.01]
  ↓
ForEach(RandomBandRotation)   [global ±1 channel rotation per band]
  ↓
TemporalAlignmentJitter       [±60ms left/right band shift]
  ↓
ElectrodePermutation   [NEW: swap up to 3 adjacent channel pairs, p=0.3]
  ↓
LogSpectrogram   [(T, 2, 16) → (T, 2, 16, 33)]
  ↓
SpecAugment      [3 time masks × 2 freq masks]
```

Val/Test: `ToTensor → LogSpectrogram` only (unchanged from all other pipelines).

---

## New Transforms

### `SoftChannelAttenuation`

**Replaces**: `ChannelDropout` + `TemporalChannelDropout`

**Why the originals fail**: Both variants zero target channels entirely. In a 16-electrode array, each channel encodes the activation of a specific forearm muscle group. Zeroing a channel removes every bit of information it carries — the model must classify keystrokes from incomplete spatial evidence that will not be missing at evaluation time. This creates a permanent train/eval distribution mismatch.

**What this does instead**: With probability `p=0.1` per channel, multiplies that channel's signal by a random factor drawn from uniform(0.3, 0.8). The channel remains present and its relative temporal pattern is fully preserved — only its amplitude is reduced. This models a real physical phenomenon: electrode contact degradation from sweat or insufficient pressure raises contact impedance, attenuating the recorded signal without eliminating it.

**Key property**: At factor 0.3×, the channel carries 30% of its normal amplitude. Relative activations across time are unchanged — the model can still discriminate between different keystrokes on that channel, just with lower SNR. This is qualitatively different from zeroing.

**Parameters**:
- `p=0.1`: 10% of channels attenuated per sample (~1–2 of 16 channels on average)
- `min_factor=0.3, max_factor=0.8`: attenuation range (30–80% of original amplitude)

---

### `AmplitudeEnvelope`

**Novel addition**: no equivalent in existing pipelines.

**Why it's needed**: sEMG signal amplitude is not stationary within a session. Muscle fatigue, perspiration accumulation, and electrode impedance drift all produce slow, session-long trends in recorded amplitude. The existing augmentations address fast, local noise but ignore this slow temporal structure. A model trained on i.i.d. samples with no temporal drift will not have learned amplitude invariance at the session timescale.

**What this does**: Samples `n_control_points=4` values uniformly from [1-max_mod, 1+max_mod] and interpolates them to the full window length T, producing a smooth envelope curve. The signal is then multiplied by this curve. Since the envelope varies slowly (4 control points over ~4 seconds of data), it operates at a timescale much longer than individual keystrokes (~100–200ms) — it cannot distort keystroke boundaries or CTC alignment.

**Key property**: Multiplicative and smooth. The relative amplitude between adjacent channels at any instant is unchanged — spatial discriminability is fully preserved. The envelope adds variability in overall signal level over time, which is exactly the real-world phenomenon it models.

**Parameters**:
- `p=0.5`: applied to half of all training samples
- `max_modulation=0.3`: envelope varies ±30% around 1.0
- `n_control_points=4`: smooth, low-frequency modulation

---

### `ElectrodePermutation`

**Complements**: `RandomBandRotation`

**Why the existing transform is insufficient**: `RandomBandRotation` shifts all channels globally by ±1 position. This is a good model of a systematic placement shift (the entire electrode array moved slightly). But inter-session placement variation often manifests as a few specific adjacent electrodes being swapped — a slightly different tightening of the armband can flip two neighboring electrodes over their respective muscle targets.

**What this does**: With probability `p=0.3`, picks 1–3 random adjacent channel pairs (i, i+1) and swaps them. This creates a varied, local permutation of the channel ordering. Across many training samples, the model learns that adjacent channels may appear in either order, building robustness to the most common form of placement variation without disrupting the global spatial structure.

**Key property**: Fully information-preserving. No signal is created or destroyed — every original sample is a valid permutation of another. Unlike `ChannelDropout`, the model receives all discriminative information; it is simply reordered.

**Parameters**:
- `p=0.3`: applied to 30% of training samples
- `max_swaps=3`: up to 3 adjacent pairs swapped per sample

---

## What Was Excluded and Why

| Transform (from `plus`) | Reason Excluded |
|-------------------------|-----------------|
| `ChannelDropout` | Replaces with `SoftChannelAttenuation` — attenuation preserves discriminative information that zeroing destroys |
| `TemporalChannelDropout` | Same root problem: zeroes temporal regions of specific channels |
| `RandomBurstNoise` | Creates high-amplitude artifacts not present in evaluation data; distribution mismatch |
| `RandomTimeWarp` | Distorts keystroke burst timing; CTC decoding assumes correct temporal ordering of brief activations |
| `RandomFrequencyEQ` | Speculative benefit in log-spectrogram domain; SpecAugment already provides frequency-domain variability |

---

## Expected Behavior

The calibrated pipeline should:
- **Outperform `log_spectrogram_plus`** by eliminating the four failure modes identified in the analysis.
- **Match or exceed `log_spectrogram` baseline** by adding meaningful variability (amplitude drift, placement variation) that the baseline does not model.
- **Generalize better across sessions** because `AmplitudeEnvelope` and `ElectrodePermutation` explicitly target inter-session variability that the baseline ignores.

If the calibrated pipeline still underperforms the baseline, the most likely explanation is that the baseline's simple augmentations (band rotation + jitter + SpecAugment) are already optimal for this dataset size, and additional augmentation of any kind adds more noise than signal. In that case, the correct response is ablating individual transforms rather than adding more.
