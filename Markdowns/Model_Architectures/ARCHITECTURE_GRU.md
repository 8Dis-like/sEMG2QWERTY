# Architecture: CNN/RNN Hybrid Model (Bi-directional GRU)

**Source file:** `emg2qwerty/cnn_rnn_hybrid.py`
**Config file:** `config/model/cnn_rnn_ctc.yaml`
**Lightning module:** `CNNRNNCTCModule`

---

## 1. Project Context & Problem Statement

The sEMG2QWERTY project aims to decode **surface electromyography (sEMG) signals** into typed text. Participants wear electrode cuffs on both forearms that measure the electrical activity of muscles during typing. The model must map a continuous stream of multi-channel EMG signals to the sequence of keys the user pressed — without knowing exactly *when* in the signal each keypress happened.

**Why this is hard:**
- EMG signals are noisy and vary significantly across users, sessions, and cuff placement.
- The 2 kHz signal is long (8,000 samples per 4-second window), making direct sequence modelling expensive.
- There is no frame-level label alignment — we only know the *sequence* of characters typed, not their exact timestamps.
- The electrode cuff may be rotated slightly on the arm between sessions, so the spatial ordering of electrode channels is not fixed.

This model builds on the baseline **TDS-Conv** architecture by inserting a **Bi-directional GRU** between the CNN encoder and the output head, combining the local pattern-detection strengths of CNNs with the long-range temporal memory of recurrent networks.

---

## 2. Input Signal & Data Pipeline

### 2.1 Raw Signal

| Property | Value |
| :--- | :--- |
| Sampling rate | 2,000 Hz |
| Electrode channels per arm | 16 |
| Arms (bands) | 2 (left, right) |
| Window length | 8,000 samples = 4 seconds |
| Padding (train/val) | 1,800 samples past + 200 samples future context |

The raw EMG data is stored in HDF5 files. Each session is a structured numpy array with fields `emg_left` and `emg_right`, each of shape `(T, 16)`.

### 2.2 Transform Pipeline

Transforms are applied before feeding data to the model. Training uses augmentation; validation and testing do not.

**Step 1 — `ToTensor`:**
```
numpy structured array → torch.Tensor of shape (T, 2, 16)
```
Reads `emg_left` and `emg_right`, converts each to float32, and stacks them along `dim=1` (the band dimension).

**Step 2 (train only) — `RandomBandRotation`:**
```
(T, 2, 16) → (T, 2, 16)
```
For each band independently, randomly rolls the 16 electrode channels by an offset sampled from `{-1, 0, +1}`. This simulates the physical uncertainty in cuff placement (the cuff might be rotated by one electrode spacing). Applied `ForEach` band independently.

**Step 3 (train only) — `TemporalAlignmentJitter`:**
```
(T, 2, 16) → (T - |offset|, 2, 16)
```
The left and right arm signals may not be perfectly aligned in time due to hardware. This augmentation shifts one band relative to the other by a random offset sampled from `[-120, +120]` samples (±60 ms at 2 kHz), then trims both to the same length.

**Step 4 — `LogSpectrogram` (n_fft=64, hop_length=16):**
```
(T, 2, 16) → (T_spec, 2, 16, 33)
```
- Applies a Short-Time Fourier Transform (STFT) to each electrode channel.
- `n_fft=64`: window size of 64 samples (32 ms at 2 kHz).
- `hop_length=16`: step of 16 samples — **effectively downsamples the signal from 2,000 Hz to 125 Hz**.
- `normalized=True, center=False`: normalizes the FFT, no zero-padding at edges.
- `freq_bins = n_fft // 2 + 1 = 33`.
- Takes `log10(spec + 1e-6)` for numerical stability and perceptual scaling.
- For an 8,000-sample window: `T_spec = (8000 - 64) / 16 + 1 = 497` frames.

**Step 5 (train only) — `SpecAugment` (n_time_masks=3, time_mask_param=25, n_freq_masks=2, freq_mask_param=4):**
```
(T_spec, 2, 16, 33) → (T_spec, 2, 16, 33)
```
Randomly zeros out rectangular regions along the time axis (up to 25 frames ≈ 200 ms) and frequency axis (up to 4 bins). Prevents overfitting on specific time-frequency patterns.

### 2.3 Final Input Shape to Model

```
(T, N, 2, 16, 33)
```
- `T` ≈ 497 (time steps in spectrogram frames, at 125 Hz)
- `N` = batch size (default 32 during training)
- `2` = number of bands (left arm, right arm)
- `16` = electrode channels per band
- `33` = frequency bins from the log-spectrogram

The key quantity `in_features = 16 × 33 = 528` is the flattened spatial+frequency feature size fed to the MLP frontend per band per time step.

---

## 3. Model Architecture

The model is assembled as a `nn.Sequential` pipeline inside `CNNRNNCTCModule.__init__`. The full pipeline is:

```
Input (T, N, 2, 16, 33)
   ↓
[1] SpectrogramNorm
   ↓
[2] MultiBandRotationInvariantMLP
   ↓
[3] nn.Flatten(start_dim=2)
   ↓
[4] CNNRNNEncoder
      ├── TDSConvEncoder (CNN)
      └── nn.GRU + projection (RNN)
   ↓
[5] nn.Linear → nn.LogSoftmax
   ↓
Output (T_out, N, num_classes)
```

---

### 3.1 SpectrogramNorm

**Class:** `SpectrogramNorm(channels=32)`
**PyTorch primitive:** `nn.BatchNorm2d(32)`

**Purpose:** Normalize the log-spectrogram features so that varying EMG signal amplitudes across users, sessions, and electrode contact quality do not dominate the learned features. Each of the 32 electrode channels (2 bands × 16 electrodes) is normalized independently.

**Why BatchNorm2d:** The spectrogram is a 2D signal with `(freq, time)` as spatial dimensions. BatchNorm2d computes normalization statistics over `(N, freq, time)` for each channel, which is exactly what we want — independent running mean/variance per electrode.

**Tensor reshaping (forward pass):**
```
Input:  (T, N, 2, 16, 33)
→ movedim(0, -1): (N, 2, 16, 33, T)
→ reshape(N, 32, 33, T)          # merge bands×channels → 32 channel groups
→ BatchNorm2d(32)                 # normalize over (N, 33, T) per channel
→ reshape(N, 2, 16, 33, T)
→ movedim(-1, 0): (T, N, 2, 16, 33)
```

**Output shape:** `(T, N, 2, 16, 33)` (unchanged, values normalized)

**Learnable parameters:** 32 scale (γ) + 32 bias (β) parameters.

---

### 3.2 MultiBandRotationInvariantMLP

**Class:** `MultiBandRotationInvariantMLP(in_features=528, mlp_features=[384], num_bands=2)`

This stage converts each band's per-timestep raw spectrogram features `(16 channels × 33 freq bins = 528 values)` into a compact embedding of size 384, while being robust to small rotations of the electrode cuff.

It contains **two independent `RotationInvariantMLP` instances**, one per band (left arm, right arm). The two MLPs do **not** share weights, allowing them to learn arm-specific patterns.

**Forward pass:**
```
Input:  (T, N, 2, 16, 33)
→ unbind(dim=2): two tensors of shape (T, N, 16, 33)
→ each through its RotationInvariantMLP → (T, N, 384)
→ stack(dim=2): (T, N, 2, 384)
```

**Output shape:** `(T, N, 2, 384)`

#### 3.2.1 RotationInvariantMLP (inner module)

**Class:** `RotationInvariantMLP(in_features=528, mlp_features=[384], pooling="mean", offsets=(-1, 0, 1))`

**Problem it solves:** The EMG electrode cuff is placed on the forearm by wrapping it around the arm. In different sessions, the cuff may be rotated by one or two electrode spacings (each electrode is ~22.5° apart on the ring). A regular MLP would learn position-specific features that fail when the cuff rotates.

**Solution:** Compute the MLP output for three rotational positions (`offset = -1, 0, +1` electrode positions), then average the results. This makes the embedding invariant to small rotational shifts.

**Network structure:**
```
Linear(528 → 384) → ReLU
```
(For `mlp_features=[384]`, this is a single-layer MLP with ReLU. For multi-layer, each element of `mlp_features` adds one Linear+ReLU block.)

**Forward pass step-by-step:**
```
Input:  (T, N, 16, 33)

Step 1 — Create rotated copies (3 offsets: -1, 0, +1):
  torch.roll(x, offset=-1, dims=2): rolls electrode dim left by 1
  torch.roll(x, offset= 0, dims=2): original
  torch.roll(x, offset=+1, dims=2): rolls electrode dim right by 1
  → torch.stack([...], dim=2): (T, N, 3, 16, 33)

Step 2 — Flatten spatial+freq features:
  x.flatten(start_dim=3): (T, N, 3, 16*33) = (T, N, 3, 528)

Step 3 — Apply MLP to each rotated version (shared weights):
  Linear(528 → 384) + ReLU: (T, N, 3, 384)

Step 4 — Pool over the 3 rotations (mean):
  x.mean(dim=2): (T, N, 384)
```

**Output shape:** `(T, N, 384)` per band

**Why mean pooling:** Averaging the outputs of shifted versions creates a smooth, rotation-invariant embedding. Max pooling would be more sensitive to the dominant rotation but less smooth. Mean pooling is the default and was found to work well empirically.

**Total learnable parameters (per band):** Linear(528 → 384) = 528×384 + 384 = 203,136 + 384 = 203,520

---

### 3.3 Flatten

```python
nn.Flatten(start_dim=2)
```

```
Input:  (T, N, 2, 384)
→ Flatten dims 2 and 3: (T, N, 768)
```

Concatenates the left and right band embeddings into a single feature vector of size `num_features = 2 × 384 = 768` per time step.

---

### 3.4 TDSConvEncoder (CNN Backbone)

**Class:** `TDSConvEncoder(num_features=768, block_channels=[24, 24, 24, 24], kernel_width=32)`

This is the convolutional backbone based on **Time-Depth Separable (TDS) convolutions** from [Hannun et al., 2019](https://arxiv.org/abs/1904.02619). It extracts local temporal patterns from the 125 Hz spectrogram feature stream.

The encoder stacks **4 TDS blocks**, each consisting of a `TDSConv2dBlock` followed by a `TDSFullyConnectedBlock`. The block configuration `[24, 24, 24, 24]` means all 4 convolutional blocks use 24 channels.

**Why TDS over standard 1D convolutions:**
- A standard 1D temporal conv over 768 features would require huge kernels or many layers to build up a meaningful receptive field.
- TDS separates the convolution into a **depth** component (channels) and a **temporal** component, reducing parameter count while maintaining expressive power.
- The "width" of the 2D conv is the spatial/feature dimension; the "length" is time.

**Total T reduction:** Each `TDSConv2dBlock` with `kernel_width=32` reduces the time axis by `kernel_width - 1 = 31` steps (no temporal padding). With 4 blocks: **total reduction = 4 × 31 = 124 frames**.

For T_spec = 497: T_cnn = 497 − 124 = **373 frames** output from the CNN.

#### 3.4.1 TDSConv2dBlock

**Class:** `TDSConv2dBlock(channels=24, width=32, kernel_width=32)`

The `width = num_features // channels = 768 // 24 = 32`.

**Purpose:** Perform a depthwise-like temporal convolution that operates independently on each of the 24 channel groups, across the 32 feature dimensions within each group, over time.

**The convolution:** `nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(1, 32))`
- Height dimension = 1 (operates across the full width of each channel in one shot)
- Width dimension = 32 (temporal kernel)
- No padding → temporal dimension shrinks by `kernel_size - 1 = 31`

**Forward pass:**
```
Input:  (T_in, N, 768)

Step 1 — Reshape for 2D conv:
  movedim(0, -1): (N, 768, T_in)
  reshape(N, 24, 32, T_in): split 768 features into (channels=24, width=32)
  → shape: (N, 24, 32, T_in)

Step 2 — Conv2d(24, 24, kernel=(1, 32)):
  Convolves along the time axis (dim 3) with kernel of size 32, no padding
  → shape: (N, 24, 32, T_in - 31)

Step 3 — ReLU activation

Step 4 — Reshape back to TNC format:
  reshape(N, 768, T_out): merge back to 768 features
  movedim(-1, 0): (T_out, N, 768)   where T_out = T_in - 31

Step 5 — Skip (residual) connection:
  x = x + inputs[-T_out:]   # align by taking last T_out frames of input
  (The input is sliced from the end because the conv is causal-aligned to the
   right edge of the kernel, so the output corresponds to the last T_out frames)

Step 6 — LayerNorm(768)

Output: (T_out, N, 768)   T_out = T_in - 31
```

**Why the skip connection works despite size mismatch:** The 2D conv without padding produces an output that is shorter than the input by `kernel_width - 1` frames. The residual connection aligns by taking `inputs[-T_out:]` — this slices off the first 31 frames of the input and adds them to the output. This is equivalent to a causal skip connection aligned to the "present" (rightmost) frame of each convolution window.

#### 3.4.2 TDSFullyConnectedBlock

**Class:** `TDSFullyConnectedBlock(num_features=768)`

**Purpose:** A pointwise (per-time-step) feedforward transformation applied after each convolution. This is analogous to the position-wise FFN in a Transformer. It mixes information across the 768 feature dimensions without any temporal mixing.

**Structure:**
```
Linear(768 → 768) → ReLU → Linear(768 → 768)
```
Followed by a residual connection and LayerNorm:
```
x = x + inputs
x = LayerNorm(768)(x)
```

**No time-axis reduction** occurs here — output shape is identical to input shape.

**Why this block follows every conv block:** The TDS conv operates on a factored (channel, width) representation and re-merges it. The FC block then re-mixes these features in the full 768-dim space before the next conv block.

---

### 3.5 CNNRNNEncoder — Bi-directional GRU

**Class:** `CNNRNNEncoder(num_features=768, block_channels=[24,24,24,24], kernel_width=32, rnn_hidden_size=384, rnn_num_layers=2, rnn_bidirectional=True)`

This class wraps both the `TDSConvEncoder` and the GRU into a single encoder module.

After the CNN stage, the features at each time step capture **local** patterns in the EMG signal (e.g., the "spike" shape of a single finger press). However, typing is a sequential activity where context matters — a "shift" key press changes what comes next, and common bigrams like "th" have predictable transitions. A CNN with a fixed receptive field cannot model these long-range dependencies. The GRU adds an **unbounded recurrent memory** that can capture patterns across the entire 4-second window.

#### 3.5.1 Why GRU?

- **vs. LSTM:** A GRU has two gates (reset gate, update gate) vs. LSTM's three (input, forget, output gates) plus a separate cell state. GRU achieves similar performance on many tasks with fewer parameters (approximately 25% fewer than LSTM for the same hidden size) and faster training.
- **vs. vanilla RNN:** GRU's gating mechanism solves the vanishing gradient problem, allowing learning of dependencies spanning hundreds of time steps.
- **vs. pure attention (Transformer):** Transformers have O(T²) attention complexity, which is expensive for T≈373 frames across the full batch. GRU has O(T) complexity. For this dataset size, GRU is more practical.

#### 3.5.2 Why Bi-directional?

In offline decoding (which is how this model is evaluated), the entire sequence is available before making predictions. Typing a key produces EMG signals that have both *anticipatory* (pre-press muscle preparation) and *follow-through* components. By processing the signal both forward and backward, the model can:
- Use future context to disambiguate similar muscle activations (e.g., a hand position that could be 'j' or 'k' might be resolved by the subsequent finger movement)
- Better align with CTC's soft alignment mechanism, which benefits from global sequence context

**The CTC loss is inherently non-causal** — it considers all possible alignments simultaneously — so using a non-causal (bidirectional) encoder is fully compatible and generally improves accuracy.

#### 3.5.3 GRU Internals

`nn.GRU(input_size=768, hidden_size=384, num_layers=2, bidirectional=True)`

**GRU equations (single layer, one direction):**
```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)       # update gate
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)       # reset gate
ñ_t = tanh(W_n · [r_t ⊙ h_{t-1}, x_t] + b_n)  # candidate hidden state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t   # output hidden state
```
- `x_t`: input at time t (size 768)
- `h_t`: hidden state (size 384 per direction)
- `σ`: sigmoid activation (gates values between 0 and 1)
- `tanh`: hyperbolic tangent (output between -1 and 1)
- `⊙`: element-wise multiplication

**Update gate** `z_t`: decides how much of the previous hidden state to carry forward. High z → remember old state, low z → accept new input.
**Reset gate** `r_t`: controls how much past information to forget when computing the candidate state. Low r → ignore past.

**2-layer stacking:** Layer 2 takes the output of Layer 1 as its input at each time step, allowing learning of more abstract temporal representations.

**Bidirectionality:** The GRU runs forward in time (t=0→T) and backward (t=T→0) simultaneously. Their hidden states are concatenated at each time step:
```
output_t = [h_forward_t ; h_backward_t]  → size 384 + 384 = 768
```

**Forward pass through the GRU stage:**
```
CNN output:   (T_cnn, N, 768)
GRU input:    (T_cnn, N, 768)      [PyTorch GRU uses (T, N, features) format]
GRU output:   (T_cnn, N, 768)     [hidden_size * 2 = 768 because bidirectional]
hidden state: (4, N, 384)         [num_layers * 2 directions, N, hidden_size]
                                   (hidden state is discarded)
```

#### 3.5.4 Output Projection and Normalization

After the GRU, the output has shape `(T_cnn, N, 768)` (since `768 = 384 * 2` from bidirectional). A linear projection maps it back to `num_features = 768` (in this case, no size change since `rnn_out_size = rnn_hidden_size * 2 = 768 = num_features`), followed by LayerNorm:

```python
self.out_projection = nn.Linear(768, 768)   # rnn_out_size → num_features
self.layer_norm = nn.LayerNorm(768)
```

**Why the projection:** Decouples the GRU's internal hidden size from the downstream classifier size, and re-scales the GRU outputs to work well with the subsequent linear classification head. Even when sizes match (as here), the linear layer adds a learnable affine transformation that helps gradient flow.

**Why LayerNorm (not BatchNorm):** At the output of the GRU, we apply LayerNorm (normalizing across the feature dimension for each time step independently) rather than BatchNorm (which normalizes across the batch). This is more stable for recurrent outputs, which can have varying statistics across time steps.

**CNNRNNEncoder full forward pass:**
```
Input:   (T, N, 768)
→ TDSConvEncoder → (T - 124, N, 768) = (T_cnn, N, 768)
→ GRU(768, 384, layers=2, bidir=True) → (T_cnn, N, 768)
→ Linear(768, 768) → (T_cnn, N, 768)
→ LayerNorm(768) → (T_cnn, N, 768)
Output:  (T_cnn, N, 768)
```

---

### 3.6 Output Head

```python
nn.Linear(768, num_classes)
nn.LogSoftmax(dim=-1)
```

**`num_classes = charset().num_classes`**: The number of possible output tokens. In the sEMG2QWERTY charset, this includes all printable keyboard characters plus a special CTC blank token. The blank index is `charset().null_class`.

**LogSoftmax:** Converts the raw logits into log-probabilities. The log form is used directly by `nn.CTCLoss`, which expects log-probabilities for numerical stability.

**Output shape:** `(T_cnn, N, num_classes)` — a probability distribution over all characters for each output time step.

---

## 4. Training Objective: CTC Loss

**`nn.CTCLoss(blank=charset().null_class)`**

### Why CTC?

We do not have frame-level alignment labels. We only know the sequence of characters typed during the 4-second window (e.g., "hello"). We do not know which of the ≈373 output frames corresponds to which character.

**Connectionist Temporal Classification (CTC)** marginalizes over all possible alignments between the output sequence and the target label sequence. It allows:
- Repeated characters (the model can output the same character over multiple frames)
- Blank tokens (silence / no character) between and within characters
- Any alignment that "collapses" (by removing blanks and duplicates) to the target sequence

**CTC input:** log-probabilities `(T_cnn, N, num_classes)` and the target character sequences.

**CTC output:** A scalar loss that is the negative log-likelihood of the target sequence under all possible CTC alignments.

### Emission Length Correction

The CNN encoder reduces the time dimension by `T_diff = T_input - T_output` (= 124 for the default config). The CTC loss needs to know the valid sequence length after reduction:

```python
T_diff = inputs.shape[0] - emissions.shape[0]
emission_lengths = input_lengths - T_diff
```

This accounts for the fact that padded sequences in a batch may have different valid lengths after the CNN shrinks the time axis.

---

## 5. Decoding

At inference time, the log-softmax outputs are decoded by a **CTC greedy decoder** (default) or a **CTC beam search decoder** (optional, configured in `config/decoder/`).

**Greedy decoding:** At each time step, take the argmax character. Then collapse the resulting sequence by removing consecutive duplicates and blank tokens.

**Beam search decoding:** Maintains a beam of top-k candidate sequences, optionally with a language model score. Produces higher accuracy at higher computational cost.

---

## 6. Optimizer and Learning Rate Schedule

**Optimizer:** Adam (`config/optimizer/adam.yaml`)

**LR Scheduler:** Linear warmup + cosine annealing (`config/lr_scheduler/linear_warmup_cosine_annealing.yaml`)
- Linearly increases the learning rate from 0 during the warmup phase.
- Then follows a cosine decay to near zero over the remaining epochs.
- This prevents early instability in the GRU's recurrent weights while ensuring effective convergence.

**Training duration:** 150 epochs maximum, with `ModelCheckpoint` monitoring `val/CER` (Character Error Rate).

---

## 7. Metrics

Three metrics are tracked across all phases (train/val/test):
- **CER** (Character Error Rate): Edit distance between predicted and ground-truth strings, normalized by target length. Primary metric.
- **IER** (Insertion Error Rate): Fraction of errors due to extra characters.
- **DER** (Deletion Error Rate): Fraction of errors due to missing characters.
- **SER** (Substitution Error Rate): Fraction of errors due to wrong characters.

---

## 8. Complete Data Flow (Default Hyperparameters)

| Stage | Module | Operation | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Raw signal | — | 2kHz EMG sampling | `(8000, 2, 16)` per sample | 4 seconds, 2 arms, 16 ch |
| ToTensor | Transform | Structured array → float32 | `(8000, 2, 16)` | |
| LogSpectrogram | Transform | STFT, hop=16 | `(497, 2, 16, 33)` | Downsampled to 125 Hz |
| **Model input** | — | After collation | `(T≈497, N, 2, 16, 33)` | T varies by batch |
| SpectrogramNorm | BN2d(32) | Per-channel normalization | `(T, N, 2, 16, 33)` | 32 channels = 2×16 |
| RotationInvariantMLP | MLP(528→384) | Rot-aug + MLP per band | `(T, N, 2, 384)` | Shared within band |
| Flatten | — | Concat bands | `(T, N, 768)` | 768 = 2 × 384 |
| TDSConv2dBlock ×4 | Conv2d(24,24,(1,32)) | Local temporal conv | `(T-31, N, 768)` per block | Shrinks by 31 each |
| TDSFCBlock ×4 | Linear ×2 | Pointwise mixing | same shape | No T reduction |
| **After CNN** | — | — | `(T-124, N, 768)` | T_cnn ≈ 373 |
| GRU | BiGRU(768→384×2, L=2) | Recurrent sequence modelling | `(T_cnn, N, 768)` | 384 per direction |
| out_projection | Linear(768→768) | Re-scale GRU outputs | `(T_cnn, N, 768)` | |
| LayerNorm | LN(768) | Feature normalization | `(T_cnn, N, 768)` | |
| Linear head | Linear(768→C) | Character logits | `(T_cnn, N, num_classes)` | |
| LogSoftmax | — | Log-probabilities | `(T_cnn, N, num_classes)` | |
| CTC Loss | — | Alignment-free seq loss | scalar | Blank = null_class |

---

## 9. Configuration & Hyperparameters

All hyperparameters for reproduction are specified in `config/model/cnn_rnn_ctc.yaml`:

```yaml
module:
  _target_: emg2qwerty.cnn_rnn_hybrid.CNNRNNCTCModule
  in_features: 528          # = 16 electrodes × 33 freq bins (n_fft//2+1)
  mlp_features: [384]       # MLP hidden/output size (single layer)
  block_channels: [24, 24, 24, 24]   # TDS conv channels per block (4 blocks)
  kernel_width: 32          # TDS temporal kernel size; each block shrinks T by 31
  rnn_hidden_size: 384      # GRU hidden units per direction
  rnn_num_layers: 2         # Number of stacked GRU layers
  rnn_bidirectional: true   # Use forward + backward GRU passes

datamodule:
  window_length: 8000       # 4-second windows at 2 kHz
  padding: [1800, 200]      # 900 ms past context + 100 ms future context
```

**Global training config** (`config/base.yaml`):
```yaml
seed: 1501
batch_size: 32
max_epochs: 150
optimizer: Adam
lr_scheduler: linear_warmup_cosine_annealing
decoder: ctc_greedy
monitor_metric: val/CER
```

**LogSpectrogram parameters** (`config/transforms/log_spectrogram.yaml`):
```yaml
logspec:
  n_fft: 64          # FFT window = 32 ms at 2 kHz; freq bins = 64//2+1 = 33
  hop_length: 16     # Frame step = 8 ms → output rate = 125 Hz
```

---

## 10. Model Parameter Count (Approximate)

| Component | Parameters |
| :--- | :--- |
| SpectrogramNorm (BN2d) | 32 × 2 = 64 |
| RotationInvariantMLP (left band) | 528×384 + 384 = 203,520 |
| RotationInvariantMLP (right band) | 528×384 + 384 = 203,520 |
| TDSConv2dBlock × 4 | 4 × (24×24×32 + 24) = ~73,728 |
| TDSFullyConnectedBlock × 4 | 4 × (768×768 + 768 + 768×768 + 768) ≈ 4.7M |
| GRU (2 layers, bidirectional) | ~4× hidden_size² × ... ≈ 5.3M |
| out_projection (Linear 768→768) | 768×768 + 768 = 590,592 |
| Final Linear (768→num_classes) | 768×num_classes |
| **Total** | **~11–12 M** (varies by num_classes) |

---

## 11. How to Reproduce / Instantiate

**Minimal Python instantiation:**

```python
from emg2qwerty.cnn_rnn_hybrid import CNNRNNCTCModule

model = CNNRNNCTCModule(
    in_features=528,            # 16 * (64//2+1)
    mlp_features=[384],
    block_channels=[24, 24, 24, 24],
    kernel_width=32,
    rnn_hidden_size=384,
    rnn_num_layers=2,
    rnn_bidirectional=True,
    optimizer={"_target_": "torch.optim.Adam", "lr": 1e-3},
    lr_scheduler={"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR"},
    decoder={"_target_": "emg2qwerty.decoder.GreedyCTCDecoder"},
)
```

**Via Hydra (standard training):**
```bash
python train.py model=cnn_rnn_ctc
```

**Input tensor format for forward pass:**
```python
# x: (T, N, num_bands=2, electrode_channels=16, freq_bins=33)
x = torch.randn(497, 8, 2, 16, 33)
log_probs = model(x)  # → (T_out ≈ 373, 8, num_classes)
```

---

## 12. Key Design Choices Summary

| Choice | Rationale |
| :--- | :--- |
| Log-spectrogram (not raw signal) | Reduces sequence length 16× (2kHz→125Hz); frequency content is highly informative for EMG |
| BatchNorm2d per electrode channel | Normalizes across users/sessions; per-channel independence is critical for multi-user generalization |
| Rotation-invariant MLP | Electrode cuff placement is unreliable; pooling over ±1 rotation makes features placement-agnostic |
| Separate MLPs per band | Left and right arm have distinct muscle groups and firing patterns; independent weights capture arm-specific features |
| TDS over standard 1D conv | Factored 2D conv is parameter-efficient; the (channel, width) factoring matches the electrode/feature structure |
| 4 TDS blocks, kernel_width=32 | Each block sees ~256ms of history (32 frames × 8ms/frame); 4 blocks → ~1 second total receptive field |
| GRU over LSTM | Similar performance, ~25% fewer parameters, faster training on limited sEMG data |
| Bidirectional GRU | Offline decoding; future context resolves ambiguity; compatible with CTC loss |
| 2-layer GRU, hidden_size=384 | Matches the embedding dimension from the CNN (768 / 2 directions = 384); deeper captures more abstract patterns |
| CTC loss | No frame-level alignment labels available; CTC handles soft alignment during training |
| Output projection + LayerNorm | Stabilizes GRU output scale before classification; LayerNorm preferred over BatchNorm for recurrent outputs |
