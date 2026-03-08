# Architecture: CNN/RNN Hybrid Model (Bi-directional LSTM)

**Source file:** `emg2qwerty/cnn_lstm_model.py`
**Config file:** `config/model/cnn_lstm_ctc.yaml`
**Lightning module:** `CNNLSTMCTCModule`

---

## 1. Project Context & Problem Statement

The sEMG2QWERTY project aims to decode **surface electromyography (sEMG) signals** into typed text. Participants wear electrode cuffs on both forearms that measure the electrical activity of muscles during typing. The model must map a continuous stream of multi-channel EMG signals to the sequence of keys the user pressed — without knowing exactly *when* in the signal each keypress happened.

**Why this is hard:**
- EMG signals are noisy and vary significantly across users, sessions, and cuff placement.
- The 2 kHz signal is long (8,000 samples per 4-second window), making direct sequence modelling expensive.
- There is no frame-level label alignment — we only know the *sequence* of characters typed, not their exact timestamps.
- The electrode cuff may be rotated slightly on the arm between sessions, so the spatial ordering of electrode channels is not fixed.

The CNN/RNN LSTM model is a **direct architectural variant** of the CNN/RNN GRU model (`cnn_rnn_hybrid.py`), with **one targeted change**: the recurrent unit is replaced from a Gated Recurrent Unit (GRU) to a **Long Short-Term Memory (LSTM)** network. All other components — the frontend, CNN backbone, output head, and training procedure — are identical.

This document describes the full architecture in detail (including the shared components) and highlights the specific differences introduced by the LSTM.

---

## 2. Input Signal & Data Pipeline

The data pipeline is **identical** to the GRU model. It is reproduced here for completeness.

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

**Step 1 — `ToTensor`:**
```
numpy structured array → torch.Tensor of shape (T, 2, 16)
```
Reads `emg_left` and `emg_right`, converts each to float32, and stacks them along `dim=1`.

**Step 2 (train only) — `RandomBandRotation`:**
```
(T, 2, 16) → (T, 2, 16)
```
For each band independently, randomly rolls the 16 electrode channels by an offset from `{-1, 0, +1}` to simulate uncertain cuff placement.

**Step 3 (train only) — `TemporalAlignmentJitter`:**
```
(T, 2, 16) → (T - |offset|, 2, 16)
```
Randomly shifts one arm's signal by up to ±120 samples (±60 ms at 2 kHz) relative to the other.

**Step 4 — `LogSpectrogram` (n_fft=64, hop_length=16):**
```
(T, 2, 16) → (T_spec, 2, 16, 33)
```
- `n_fft=64`: 32 ms window; `freq_bins = 64//2 + 1 = 33`.
- `hop_length=16`: 8 ms step → **downsamples from 2,000 Hz to 125 Hz**.
- `normalized=True, center=False`.
- `log10(spec + 1e-6)` for stability.
- For 8,000 samples: `T_spec = (8000 - 64) / 16 + 1 = 497` frames.

**Step 5 (train only) — `SpecAugment`:**
```
(T_spec, 2, 16, 33) → (T_spec, 2, 16, 33)
```
Randomly zeros time (up to 25 frames ≈ 200 ms) and frequency (up to 4 bins) regions.

### 2.3 Final Input Shape to Model

```
(T, N, 2, 16, 33)
```
- `T` ≈ 497 (spectrogram frames at 125 Hz)
- `N` = batch size (default 32)
- `2` = number of bands (left/right arm)
- `16` = electrode channels per band
- `33` = frequency bins
- `in_features = 16 × 33 = 528` (flattened per band per time step)

---

## 3. Model Architecture

The model is assembled as a `nn.Sequential` pipeline inside `CNNLSTMCTCModule.__init__`. The structure is:

```
Input (T, N, 2, 16, 33)
   ↓
[1] SpectrogramNorm
   ↓
[2] MultiBandRotationInvariantMLP
   ↓
[3] nn.Flatten(start_dim=2)
   ↓
[4] CNNLSTMEncoder
      ├── TDSConvEncoder (CNN)
      └── nn.LSTM + projection  ← PRIMARY DIFFERENCE FROM GRU MODEL
   ↓
[5] nn.Linear → nn.LogSoftmax
   ↓
Output (T_out, N, num_classes)
```

Stages 1–3, the TDS CNN backbone, and the output head are **identical** to the GRU model. The only architectural difference is stage 4's recurrent unit: `nn.LSTM` instead of `nn.GRU`.

---

### 3.1 SpectrogramNorm

**Class:** `SpectrogramNorm(channels=32)`
**PyTorch primitive:** `nn.BatchNorm2d(32)`

Normalizes each of the 32 electrode channels (2 bands × 16) independently using 2D Batch Normalization. This addresses cross-user and cross-session amplitude variability in EMG signals.

**Tensor reshaping:**
```
Input:  (T, N, 2, 16, 33)
→ movedim(0, -1): (N, 2, 16, 33, T)
→ reshape(N, 32, 33, T)
→ BatchNorm2d(32)              # normalize over (N, freq, T) per channel
→ reshape(N, 2, 16, 33, T)
→ movedim(-1, 0): (T, N, 2, 16, 33)
```

**Output shape:** `(T, N, 2, 16, 33)` (same shape, values normalized)
**Learnable parameters:** 64 (32 scale + 32 bias)

---

### 3.2 MultiBandRotationInvariantMLP

**Class:** `MultiBandRotationInvariantMLP(in_features=528, mlp_features=[384], num_bands=2)`

Contains two independent `RotationInvariantMLP` instances (one per band) that embed each arm's spectrogram features into a 384-dimensional vector, robust to ±1 electrode rotation.

**Forward pass:**
```
Input:  (T, N, 2, 16, 33)
→ unbind(dim=2): two tensors (T, N, 16, 33)
→ each → RotationInvariantMLP → (T, N, 384)
→ stack(dim=2): (T, N, 2, 384)
```

**Output shape:** `(T, N, 2, 384)`

#### 3.2.1 RotationInvariantMLP (inner module)

**Class:** `RotationInvariantMLP(in_features=528, mlp_features=[384], pooling="mean", offsets=(-1, 0, 1))`

Applies a shared MLP to three rotational versions of each band's electrodes, then averages the results to produce a placement-invariant embedding.

**Forward pass:**
```
Input:  (T, N, 16, 33)

Step 1 — Roll electrode channels by offsets (-1, 0, +1) and stack:
  → (T, N, 3, 16, 33)

Step 2 — Flatten spatial+freq dims (from dim 3 onwards):
  → (T, N, 3, 528)

Step 3 — Apply shared MLP [Linear(528→384) + ReLU]:
  → (T, N, 3, 384)

Step 4 — Mean-pool over rotations (dim=2):
  → (T, N, 384)
```

**Output:** `(T, N, 384)` per band
**Parameters:** 528×384 + 384 = 203,520 per band

---

### 3.3 Flatten

```python
nn.Flatten(start_dim=2)
```
```
(T, N, 2, 384) → (T, N, 768)
```
Concatenates left and right band embeddings. `num_features = 2 × 384 = 768`.

---

### 3.4 TDSConvEncoder (CNN Backbone)

**Class:** `TDSConvEncoder(num_features=768, block_channels=[24, 24, 24, 24], kernel_width=32)`

Four TDS blocks, each consisting of a `TDSConv2dBlock` followed by a `TDSFullyConnectedBlock`. Extracts local temporal patterns from the 125 Hz spectrogram feature stream.

**Total time reduction:** 4 × (32 − 1) = **124 frames**
For T_spec = 497: T_cnn = 497 − 124 = **373 output frames**

#### 3.4.1 TDSConv2dBlock

**Class:** `TDSConv2dBlock(channels=24, width=32, kernel_width=32)`

where `width = num_features // channels = 768 // 24 = 32`.

**Convolution:** `nn.Conv2d(24, 24, kernel=(1, 32))` — no padding, shrinks T by 31.

**Forward pass:**
```
Input:  (T_in, N, 768)
→ movedim(0,-1) + reshape(N, 24, 32, T_in)
→ Conv2d(24, 24, (1,32)) → (N, 24, 32, T_in-31)
→ ReLU
→ reshape(N, 768, T_out) + movedim(-1,0) → (T_out, N, 768)
→ skip: x + inputs[-T_out:]   # align residual to right edge of conv window
→ LayerNorm(768)
Output: (T_out=T_in-31, N, 768)
```

The residual connection uses `inputs[-T_out:]` because the unpadded convolution aligns outputs to the rightmost frame of each window.

#### 3.4.2 TDSFullyConnectedBlock

**Class:** `TDSFullyConnectedBlock(num_features=768)`

Pointwise (per-time-step) feedforward block that mixes across the 768 feature dimensions.

```
Linear(768→768) + ReLU + Linear(768→768)
→ skip: x + inputs
→ LayerNorm(768)
```

No time reduction. Re-mixes features in the full 768-dim space after each conv block.

---

### 3.5 CNNLSTMEncoder — Bi-directional LSTM

**Class:** `CNNLSTMEncoder(num_features=768, block_channels=[24,24,24,24], kernel_width=32, rnn_hidden_size=384, rnn_num_layers=2, rnn_bidirectional=True)`

This is the **primary architectural difference** from the GRU model. The recurrent unit is `nn.LSTM` instead of `nn.GRU`.

The motivations for adding a recurrent layer are the same as the GRU model:
- The CNN has a fixed receptive field (~1 second); LSTM adds unbounded sequence memory
- Typing is sequential; keypresses have linguistic dependencies (bigrams, shift+letter, etc.)
- Bidirectionality allows future context to resolve ambiguous muscle activations

#### 3.5.1 LSTM vs. GRU: The Core Difference

Both LSTM and GRU are gated recurrent units designed to solve the vanishing gradient problem, but they differ in internal mechanism and parameter count.

**GRU (2 gates):**
```
z_t = σ(W_z · [h_{t-1}, x_t])        # update gate
r_t = σ(W_r · [h_{t-1}, x_t])        # reset gate
ñ_t = tanh(W_n · [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t
```

**LSTM (3 gates + cell state):**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # input gate
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # forget gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # cell gate (candidate values)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t     # cell state update
h_t = o_t ⊙ tanh(c_t)                # hidden state output
```

where `x_t` is the input (size 768), `h_t` is the hidden state (size 384 per direction), and `c_t` is the **cell state** (size 384 per direction).

**The LSTM's key structural addition: the cell state `c_t`**

The LSTM maintains *two* internal states per time step:
1. **Hidden state `h_t`** — the "short-term" memory output at time t (same role as GRU's `h_t`)
2. **Cell state `c_t`** — a dedicated "long-term" memory lane that flows through time with minimal transformation

The cell state is the fundamental innovation of LSTM (Hochreiter & Schmidhuber, 1997). Information can be written to or erased from the cell state with minimal gradient degradation, because the gradient of `c_t` with respect to `c_{t-1}` is approximately `f_t` (the forget gate). If the forget gate is near 1.0, the gradient flows backward almost unchanged — enabling learning of dependencies spanning thousands of time steps.

**Gate-by-gate explanation:**

- **Forget gate `f_t`:** Controls what to erase from the cell state. `f_t ≈ 1` → keep everything; `f_t ≈ 0` → erase. For example, after a "Return" key, the model should forget the context of the previous line.

- **Input gate `i_t`:** Controls what new information to write to the cell state. Works together with the cell gate `g_t` (candidate values): `c_t += i_t ⊙ g_t`.

- **Cell gate `g_t`:** Proposes new candidate values to potentially add to the cell state. This is the "content" of what gets written.

- **Output gate `o_t`:** Controls what portion of the cell state to expose as the hidden state `h_t`. The cell state is squashed through tanh and then gated: `h_t = o_t ⊙ tanh(c_t)`.

**Comparison table: GRU vs. LSTM:**

| Property | GRU | LSTM |
| :--- | :--- | :--- |
| Number of gates | 2 (update, reset) | 3 (input, forget, output) |
| Internal states | 1 (hidden state h_t) | 2 (hidden state h_t + cell state c_t) |
| Parameters per unit | ~3× (hidden²) | ~4× (hidden²) |
| Gradient flow | Through h_t | Separate highway via c_t |
| Memory type | Single unified memory | Separated short-term (h) + long-term (c) |
| Expressiveness | Slightly less | Slightly more |
| Training speed | Faster (~25% less compute) | Slower |
| Typical advantage | Limited data, faster iteration | Complex long-range dependencies |

For EMG typing data with limited corpus size and 4-second windows, the GRU and LSTM often perform comparably. The LSTM is provided as an alternative to empirically compare whether the additional cell state helps model longer dependencies in typing patterns.

#### 3.5.2 LSTM Parameters

`nn.LSTM(input_size=768, hidden_size=384, num_layers=2, bidirectional=True)`

- `input_size = 768`: matches `num_features` from the CNN encoder output
- `hidden_size = 384`: number of hidden units per direction
- `num_layers = 2`: two stacked LSTM layers (layer 2 takes output of layer 1 as input)
- `bidirectional = True`: processes sequence both forward and backward

**Output of LSTM:**
```
LSTM output tuple: (output, (h_n, c_n))
  output: (T_cnn, N, 768)  # hidden_size * 2 directions = 384*2 = 768
  h_n:    (4, N, 384)      # num_layers * 2 directions, N, hidden_size
  c_n:    (4, N, 384)      # same shape as h_n (cell state per layer/direction)
```

Only `output` is used (via `x, _ = self.rnn(x)`). The final hidden and cell states are discarded because CTC decoding operates on the per-frame outputs, not the terminal state.

#### 3.5.3 Bidirectionality

The LSTM runs two independent passes simultaneously:
- **Forward pass:** t = 0 → T_cnn (sees past context at each step)
- **Backward pass:** t = T_cnn → 0 (sees future context at each step)

At each time step t, the outputs are concatenated:
```
output_t = [h_forward_t ; h_backward_t]   → size 768 (384 + 384)
```

The 2-layer LSTM with bidirectionality has 4 LSTM "cells" total:
- Layer 1 forward, Layer 1 backward (inputs from feature sequence)
- Layer 2 forward, Layer 2 backward (inputs from Layer 1 outputs)

**Multi-layer with bidirectionality:** In PyTorch's `nn.LSTM`, for a bidirectional 2-layer network:
- Layer 1 forward takes `x_t` as input
- Layer 1 backward takes `x_t` as input (reversed)
- Layer 2 forward takes the *concatenated* layer 1 forward/backward output as input
- Layer 2 backward takes the *concatenated* layer 1 forward/backward output as input (reversed)

This allows Layer 2 to build higher-order temporal abstractions on top of Layer 1's full bidirectional context.

#### 3.5.4 Why Bidirectional for Typing?

Typing patterns have temporal structure in both directions:
- **Anticipatory signals:** Muscles begin activating *before* a key is physically pressed (pre-motor planning, ~50–100ms). Looking backward from a keypress's EMG signature can reveal this preparatory signal.
- **Follow-through:** After a keypress, there is often a characteristic rebound/relaxation pattern. Looking forward from a potential press helps confirm it.
- **Linguistic bigrams/trigrams:** Common letter sequences (e.g., "the", "ing") create predictable EMG patterns. Bidirectional context helps distinguish ambiguous individual characters within a known sequence.

#### 3.5.5 Output Projection and Normalization

Identical to the GRU model:

```python
rnn_out_size = 384 * 2 = 768    # hidden_size * num_directions
self.out_projection = nn.Linear(768, 768)    # rnn_out_size → num_features
self.layer_norm = nn.LayerNorm(768)
```

**Purpose of the projection:** Even when `rnn_out_size == num_features` (as here), the linear layer provides a learnable affine transformation. This re-scales the LSTM's internal state representations into a space well-suited for the downstream classifier, and provides a stable interface for gradient flow.

**Why LayerNorm over BatchNorm:** LayerNorm normalizes across the feature dimension for each `(t, n)` pair independently, making it robust to variable sequence lengths and batch sizes. LSTM outputs can have high variance at certain time steps, and LayerNorm prevents this from destabilizing the classification head.

**CNNLSTMEncoder full forward pass:**
```
Input:   (T, N, 768)
→ TDSConvEncoder → (T - 124, N, 768) = (T_cnn, N, 768)
→ LSTM(768, 384, layers=2, bidir=True) → (T_cnn, N, 768)
                                           [hidden/cell states discarded]
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

`num_classes = charset().num_classes` — all printable keyboard characters plus the CTC blank token.

**Output shape:** `(T_cnn, N, num_classes)` — log-probabilities over all characters at each output frame.

---

## 4. Training Objective: CTC Loss

**`nn.CTCLoss(blank=charset().null_class)`**

Identical to the GRU model. CTC (Connectionist Temporal Classification) marginalizes over all possible alignments between the output frame sequence and the target character sequence, enabling training without frame-level labels.

**Emission length correction:**
```python
T_diff = inputs.shape[0] - emissions.shape[0]  # = 124 for default config
emission_lengths = input_lengths - T_diff
```

---

## 5. Decoding

Identical to GRU model. At inference, the log-softmax outputs are decoded via:
- **Greedy CTC decoding** (default): argmax at each frame, collapse duplicates and blanks.
- **Beam search CTC decoding** (optional): maintain top-k candidate sequences.

---

## 6. Optimizer and Learning Rate Schedule

**Identical to GRU model:**
- **Optimizer:** Adam
- **LR Scheduler:** Linear warmup + cosine annealing
- **Max epochs:** 150
- **Checkpoint metric:** `val/CER`

---

## 7. Metrics

Identical to GRU model:
- **CER** (Character Error Rate) — primary metric
- **IER** (Insertion Error Rate)
- **DER** (Deletion Error Rate)
- **SER** (Substitution Error Rate)

---

## 8. Complete Data Flow (Default Hyperparameters)

| Stage | Module | Operation | Output Shape | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Raw signal | — | 2kHz EMG sampling | `(8000, 2, 16)` per sample | 4 seconds, 2 arms, 16 ch |
| ToTensor | Transform | Structured array → float32 | `(8000, 2, 16)` | |
| LogSpectrogram | Transform | STFT, hop=16 | `(497, 2, 16, 33)` | 2kHz → 125 Hz |
| **Model input** | — | After collation | `(T≈497, N, 2, 16, 33)` | |
| SpectrogramNorm | BN2d(32) | Per-channel normalization | `(T, N, 2, 16, 33)` | 32 = 2×16 channels |
| RotationInvariantMLP | MLP(528→384) | Rot-aug + MLP per band | `(T, N, 2, 384)` | |
| Flatten | — | Concat bands | `(T, N, 768)` | 768 = 2 × 384 |
| TDSConv2dBlock ×4 | Conv2d(24,24,(1,32)) | Local temporal conv | `(T-31, N, 768)` per block | Shrinks T by 31 each |
| TDSFCBlock ×4 | Linear ×2 | Pointwise mixing | same shape | No T change |
| **After CNN** | — | — | `(T-124, N, 768)` | T_cnn ≈ 373 |
| **LSTM** | **BiLSTM(768→384×2, L=2)** | **Recurrent sequence modelling** | `(T_cnn, N, 768)` | **h_n, c_n discarded** |
| out_projection | Linear(768→768) | Re-scale LSTM outputs | `(T_cnn, N, 768)` | |
| LayerNorm | LN(768) | Feature normalization | `(T_cnn, N, 768)` | |
| Linear head | Linear(768→C) | Character logits | `(T_cnn, N, num_classes)` | |
| LogSoftmax | — | Log-probabilities | `(T_cnn, N, num_classes)` | |
| CTC Loss | — | Alignment-free seq loss | scalar | Blank = null_class |

Rows in **bold** highlight the components that differ from (or are specific to) the LSTM model.

---

## 9. Configuration & Hyperparameters

All hyperparameters for reproduction are specified in `config/model/cnn_lstm_ctc.yaml`:

```yaml
module:
  _target_: emg2qwerty.cnn_lstm_model.CNNLSTMCTCModule
  in_features: 528          # = 16 electrodes × 33 freq bins (n_fft//2+1)
  mlp_features: [384]       # MLP hidden/output size (single layer)
  block_channels: [24, 24, 24, 24]   # TDS conv channels per block (4 blocks)
  kernel_width: 32          # TDS temporal kernel size; each block shrinks T by 31
  rnn_hidden_size: 384      # LSTM hidden units per direction
  rnn_num_layers: 2         # Number of stacked LSTM layers
  rnn_bidirectional: true   # Use forward + backward LSTM passes

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

**LogSpectrogram** (`config/transforms/log_spectrogram.yaml`):
```yaml
logspec:
  n_fft: 64          # freq bins = 33
  hop_length: 16     # 2kHz → 125 Hz
```

---

## 10. Model Parameter Count (Approximate)

| Component | Parameters | vs. GRU |
| :--- | :--- | :--- |
| SpectrogramNorm (BN2d) | 64 | = same |
| RotationInvariantMLP (×2 bands) | 2 × 203,520 = 407,040 | = same |
| TDSConv2dBlocks × 4 | ~73,728 | = same |
| TDSFullyConnectedBlocks × 4 | ~4.7M | = same |
| **LSTM (2 layers, bidirectional)** | **~7.1M** | **~+1.8M more than GRU** |
| out_projection (Linear 768→768) | 590,592 | = same |
| Final Linear (768→num_classes) | 768×num_classes | = same |
| **Total** | **~13–14 M** | **~+1.8M vs. GRU** |

**Why LSTM has more parameters than GRU:**

For a single LSTM layer (one direction) with `input_size=d` and `hidden_size=h`:
- 4 weight matrices: W_i, W_f, W_g, W_o — each `h × (d+h)` → `4 × h × (d+h)` weights
- 4 bias vectors: `4 × h` biases
- Total: `4 × h × (d+h) + 4h`

For a single GRU layer (one direction):
- 3 weight matrices for z, r, n — each `h × (d+h)` + slightly different structure → `3 × h × (d+h)` weights
- Total: `3 × h × (d+h) + 3h`

At `d=768, h=384`:
- LSTM: 4 × 384 × (768+384) + 4×384 = 4 × 384 × 1152 + 1536 ≈ 1,771,008 per layer/direction
- GRU: 3 × 384 × (768+384) + 3×384 = 3 × 384 × 1152 + 1152 ≈ 1,328,640 per layer/direction
- For 2 layers, 2 directions: LSTM ≈ 14.2M for this module; GRU ≈ 10.6M

(Values are approximate; include the projection layer for exact counts.)

---

## 11. How to Reproduce / Instantiate

**Minimal Python instantiation:**

```python
from emg2qwerty.cnn_lstm_model import CNNLSTMCTCModule

model = CNNLSTMCTCModule(
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

**Via Hydra:**
```bash
python train.py model=cnn_lstm_ctc
```

**Input tensor format:**
```python
# x: (T, N, num_bands=2, electrode_channels=16, freq_bins=33)
x = torch.randn(497, 8, 2, 16, 33)
log_probs = model(x)  # → (T_out ≈ 373, 8, num_classes)
```

---

## 12. GRU vs. LSTM: When to Choose Which

| Scenario | Preferred | Reason |
| :--- | :--- | :--- |
| Limited data / fast iteration | GRU | ~25% fewer parameters, faster training |
| Longer dependencies needed | LSTM | Cell state provides a dedicated long-term memory highway |
| Debugging / ablation studies | GRU | Simpler internals, easier to reason about |
| Maximum accuracy at higher cost | LSTM | More expressive; worth trying if GRU plateaus |
| Real-time / latency-constrained | GRU | Less compute per time step |

For sEMG typing, the 4-second windows at 125 Hz produce ~373 output frames. Both GRU and LSTM are capable of modelling dependencies across this range. The performance difference is empirical and dataset-dependent — running both and comparing `val/CER` is the recommended approach.

---

## 13. Key Design Choices Summary

| Choice | Rationale |
| :--- | :--- |
| Log-spectrogram (not raw signal) | Reduces sequence 16× (2kHz→125Hz); frequency content captures motor unit recruitment patterns |
| BatchNorm2d per electrode channel | Normalizes amplitude variation across users and sessions |
| Rotation-invariant MLP | Electrode cuff placement uncertainty modelled by pooling over ±1 electrode offsets |
| Separate MLPs per band | Left and right arm muscles differ; independent weights capture arm-specific biomechanics |
| TDS CNN backbone | Parameter-efficient local temporal feature extraction; 4 blocks build ~1s receptive field |
| LSTM over GRU | Added cell state provides a separate long-term memory lane; slightly more expressive |
| Bidirectional LSTM | Offline decoding; anticipatory muscle signals and follow-through patterns require future context |
| 2-layer LSTM, hidden_size=384 | Matches CNN embedding dimension; 2 layers enable hierarchical temporal abstraction |
| CTC loss | No frame-level labels available; CTC handles automatic alignment during training |
| Linear projection + LayerNorm | Stable LSTM output re-scaling; LayerNorm is preferred over BatchNorm for recurrent outputs |
| Mean-pool over electrode rotations | Smooth, spatially invariant embedding; max-pool would be sensitive to dominant rotation |
