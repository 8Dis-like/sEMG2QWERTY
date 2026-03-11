# Architecture: Pure RNN Model (Deep Bidirectional GRU, No CNN)

**Source file:** `emg2qwerty/pure_rnn.py`
**Config file:** `config/model/pure_rnn_ctc.yaml`
**Lightning module:** `PureRNNCTCModule`

---

## 1. Project Context & Problem Statement

The sEMG2QWERTY project aims to decode **surface electromyography (sEMG) signals** into typed text. Participants wear electrode cuffs on both forearms that measure the electrical activity of muscles during typing. The model must map a continuous stream of multi-channel EMG signals to the sequence of keys the user pressed — without knowing exactly *when* in the signal each keypress happened.

**Why this is hard:**
- EMG signals are noisy and vary significantly across users, sessions, and cuff placement.
- The 2 kHz signal is long (8,000 samples per 4-second window), making direct sequence modelling expensive.
- There is no frame-level label alignment — we only know the *sequence* of characters typed, not their exact timestamps.
- The electrode cuff may be rotated slightly on the arm between sessions, so the spatial ordering of electrode channels is not fixed.

**What makes this model different:**

All other models in this codebase (TDS-Conv, CNN/RNN GRU, CNN/RNN LSTM) use a **CNN backbone** (the TDS convolutional encoder) before any recurrent processing. This model is the only one that eliminates the CNN entirely. All temporal feature extraction is performed recurrently through a **deep stack of bidirectional GRU layers**, each with residual connections and inter-layer normalization.

This is a fundamentally different inductive bias:
- CNN models build up temporal features hierarchically with a fixed local receptive field, then use an RNN to model global context on top of those features.
- The Pure RNN model immediately processes the full sequence context at every layer, from the very first step.

---

## 2. Input Signal & Data Pipeline

The data pipeline is **identical** to the CNN/RNN hybrid models. It is reproduced here for completeness.

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
Reads `emg_left` and `emg_right`, converts each to float32, and stacks them along `dim=1` (the band dimension).

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
- `log10(spec + 1e-6)` for numerical stability.
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

The model is assembled as a `nn.Sequential` pipeline inside `PureRNNCTCModule.__init__`. The full pipeline is:

```
Input (T, N, 2, 16, 33)
   ↓
[1] SpectrogramNorm
   ↓
[2] MultiBandRotationInvariantMLP
   ↓
[3] nn.Flatten(start_dim=2)
   ↓
[4] DeepBiGRUEncoder              ← REPLACES TDSConvEncoder + single GRU
      ├── (optional) Input Projection
      ├── ResidualBiGRUBlock × 4
      │     ├── nn.GRU (1 layer, bidirectional)
      │     ├── nn.Dropout
      │     ├── Residual connection
      │     └── nn.LayerNorm
      └── out_projection + LayerNorm
   ↓
[5] nn.Linear → nn.LogSoftmax
   ↓
Output (T, N, num_classes)      ← NOTE: T is unchanged (no CNN shrinkage)
```

**Key distinction from hybrid models:** The CNN/RNN hybrids reduce T from 497 to 373 (a 124-frame shrinkage from 4 TDS blocks). The pure RNN model preserves the full T ≈ 497 frames, giving the CTC decoder more output positions to work with.

---

### 3.1 SpectrogramNorm

**Class:** `SpectrogramNorm(channels=32)`
**PyTorch primitive:** `nn.BatchNorm2d(32)`

Identical to the hybrid models. Normalizes each of the 32 electrode channels (2 bands × 16) independently using 2D Batch Normalization, addressing cross-user and cross-session amplitude variability.

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

Identical to the hybrid models. Applies a separate `RotationInvariantMLP` to each band (left arm, right arm), embedding each arm's 528-dimensional per-timestep features into a 384-dimensional rotation-invariant representation.

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

Applies a shared MLP to three rotational versions of each band's electrode channels, then averages the results. This makes the embedding invariant to ±1 electrode rotation (uncertain cuff placement).

```
Input:  (T, N, 16, 33)
→ roll electrodes by {-1, 0, +1}, stack: (T, N, 3, 16, 33)
→ flatten from dim 3: (T, N, 3, 528)
→ MLP [Linear(528→384) + ReLU]: (T, N, 3, 384)
→ mean over rotations: (T, N, 384)
```

**Parameters:** 528×384 + 384 = 203,520 per band (×2 bands = 407,040 total)

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

### 3.4 DeepBiGRUEncoder — The Core Module

**Class:** `DeepBiGRUEncoder(num_features=768, hidden_size=384, num_layers=4, dropout=0.2)`

This is the defining component of the Pure RNN model. It replaces both the `TDSConvEncoder` **and** the single `nn.GRU` of the hybrid models with a unified deep bidirectional GRU stack.

#### 3.4.1 Design Rationale: Why Deep RNN Without CNN?

In the CNN/RNN hybrid, the CNN's role is twofold:
1. **Local feature extraction** — detect the "shape" of individual EMG spikes across a 256 ms window.
2. **Temporal compression** — reduce T by 124 frames, focusing the RNN on a manageable sequence.

Removing the CNN forces the RNN to perform both tasks. To compensate:

- **Depth (4 layers):** Each layer can specialize — lower layers capture short-range patterns; higher layers aggregate over longer contexts. This mirrors how deep CNNs build hierarchical features across layers.
- **Bidirectionality (both directions):** Every GRU layer sees the full forward and backward context, making the representations immediately global.
- **Residual connections (per layer):** Prevents degradation across 4 layers; gradients flow directly to any layer.
- **No time shrinkage:** Preserves all ~497 frames, giving CTC more alignment options (beneficial when characters are spread unevenly over the window).

#### 3.4.2 Input Projection

With `num_features = 768` and `hidden_size = 384`, the GRU output size is `384 × 2 = 768 = num_features`. This equality means the residual connections in each block require no extra parameters. The input projection is therefore an `nn.Identity()` — a no-op — in the default configuration.

If the hidden size were changed such that `hidden_size * 2 ≠ num_features`, an automatic `Linear(num_features, hidden_size*2) + LayerNorm` projection is inserted here.

```
Input:  (T, N, 768)
→ Identity (no-op, sizes already match)
→ (T, N, 768)
```

#### 3.4.3 ResidualBiGRUBlock (×4)

**Class:** `ResidualBiGRUBlock(input_size=768, hidden_size=384, dropout=0.2)`

Each block wraps a **single bidirectional GRU layer** with three stabilizing components:

**GRU equations (single direction, one layer):**
```
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)            # update gate
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)            # reset gate
ñ_t = tanh(W_n · [r_t ⊙ h_{t-1}, x_t] + b_n)  # candidate state
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ñ_t         # output hidden state
```
- `x_t`: input (size 768)
- `h_t`: hidden state (size 384 per direction)
- `σ`: sigmoid; `tanh`: hyperbolic tangent; `⊙`: element-wise multiply

The block runs the GRU in **both directions** simultaneously. At each time step, forward and backward hidden states are concatenated:
```
output_t = [h_forward_t ; h_backward_t]   → size 768
```

**Forward pass of one ResidualBiGRUBlock:**
```
Input:   (T, N, 768)

Step 1 — Bidirectional GRU:
  gru(input) → (T, N, 768)     [384 forward + 384 backward]

Step 2 — Dropout(p=0.2):
  (T, N, 768)                  [disabled on the last block]

Step 3 — Residual addition:
  x = gru_out + inputs         [direct residual, no projection needed
                                since 768 == 768]

Step 4 — LayerNorm(768):
  normalize across the feature dim for each (t, n) pair

Output:  (T, N, 768)
```

**Why the residual is applied before LayerNorm (Pre-norm style):** The residual shortcut is added to the raw GRU output before normalization, which is the convention used in the TDS blocks of the hybrid models. This allows LayerNorm to operate on the combined signal, centering and scaling the joint representation rather than only the GRU's contribution.

**Why no dropout on the last block:** The final block's output goes directly to the output projection. Applying dropout there would randomly zero features that the subsequent Linear and LayerNorm depend on, destabilizing the gradient signal to the classifier head.

**Parameters per block (one direction, input_size=hidden_size=384):**
- GRU weight matrices (3 gates): `3 × hidden_size × (input_size + hidden_size)` = `3 × 384 × 768` = 884,736
- GRU biases: `3 × 384` = 1,152
- Per direction: ~885,888; for 2 directions: ~1,771,776
- LayerNorm(768): 768 × 2 = 1,536
- **Total per block: ~1,773,312**

**Layer 1 input_size = 768 (from Flatten); Layers 2–4 input_size = 768 (output of previous block)**. All 4 blocks have identical parameter counts.

**Stacked 4-layer forward pass:**
```
Layer 1: (T, N, 768) → BiGRU → Dropout → Residual → LN → (T, N, 768)
Layer 2: (T, N, 768) → BiGRU → Dropout → Residual → LN → (T, N, 768)
Layer 3: (T, N, 768) → BiGRU → Dropout → Residual → LN → (T, N, 768)
Layer 4: (T, N, 768) → BiGRU → [no Dropout] → Residual → LN → (T, N, 768)
```

**Why 4 layers?** The CNN/RNN hybrid uses 4 TDS blocks (CNN) + 2 GRU layers = effectively 6 sequential processing stages. The pure RNN uses 4 GRU layers to achieve comparable representational depth without convolutions.

#### 3.4.4 Output Projection and Normalization

After the 4 ResidualBiGRUBlocks, the output is `(T, N, 768)`. The output projection maps from the GRU output space back to `num_features`:

```python
self.out_projection = nn.Linear(768, 768)   # output_size → num_features
self.out_norm = nn.LayerNorm(768)
```

In the default configuration `output_size = hidden_size * 2 = 768 = num_features`, so this is a square linear transformation. Its purpose is to provide a learnable re-scaling of the stacked GRU's output before it hits the classifier head — analogous to the `out_projection + layer_norm` in the hybrid models' `CNNRNNEncoder`.

**DeepBiGRUEncoder full forward pass:**
```
Input:    (T, N, 768)
→ Identity (input_proj)             → (T, N, 768)
→ ResidualBiGRUBlock × 4           → (T, N, 768)    [T unchanged]
→ Linear(768, 768)                  → (T, N, 768)
→ LayerNorm(768)                    → (T, N, 768)
Output:   (T, N, 768)
```

**No time reduction:** Unlike the TDS CNN blocks (each of which shrinks T by 31), GRUs produce an output for every input time step. T remains at ≈497 throughout the encoder.

---

### 3.5 Output Head

```python
nn.Linear(768, num_classes)
nn.LogSoftmax(dim=-1)
```

`num_classes = charset().num_classes` — all printable keyboard characters plus the CTC blank token.

**Output shape:** `(T, N, num_classes)` — log-probabilities over all characters at each of the ~497 output frames.

**Note on T:** For the hybrid models, T_out ≈ 373 (after 124-frame CNN shrinkage). For the pure RNN model, T_out ≈ 497 (full sequence). This gives CTC ~33% more output frames to distribute alignments across — a meaningful difference for short target sequences.

---

## 4. Training Objective: CTC Loss

**`nn.CTCLoss(blank=charset().null_class)`**

Identical to all other models. CTC marginalizes over all valid alignments between the output frame sequence and the target character sequence, enabling training without frame-level labels.

**Emission length computation:**
```python
T_diff = inputs.shape[0] - emissions.shape[0]  # ≈ 0 for pure RNN
emission_lengths = input_lengths - T_diff
```

For the pure RNN, `T_diff ≈ 0` (the MLP frontend does not shrink T). The formula is kept for correctness in edge cases where padding or spectrogram rounding could introduce minor differences.

**CTC and longer output sequences:** CTC requires `T_out ≥ 2L - 1` where `L` is the number of characters in the target sequence (due to mandatory blank-between-repeats). The pure RNN's ~497 output frames vs. the hybrid's ~373 makes this constraint easier to satisfy for longer words — a benefit for rare long tokens.

---

## 5. Decoding

Identical to all other models. At inference, the log-softmax outputs are decoded via:
- **Greedy CTC decoding** (default): argmax at each frame, collapse consecutive duplicates and blanks.
- **Beam search CTC decoding** (optional): maintain top-k candidate sequences, optionally with language model scoring.

---

## 6. Optimizer and Learning Rate Schedule

**Identical to the hybrid models:**
- **Optimizer:** Adam
- **LR Scheduler:** Linear warmup + cosine annealing (`linear_warmup_cosine_annealing`)
- **Max epochs:** 150
- **Checkpoint metric:** `val/CER`

The linear warmup is especially important for the pure RNN since the deeper GRU stack (4 vs 2 layers) has more recurrent weight matrices to initialize. A cold start at the full learning rate would destabilize the early training dynamics. The warmup gradually increases the learning rate, allowing the recurrent weights to settle into a reasonable regime before full-scale optimization.

---

## 7. Metrics

Identical to all other models:
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
| SpectrogramNorm | BN2d(32) | Per-channel normalization | `(T, N, 2, 16, 33)` | 32 = 2×16 |
| RotationInvariantMLP | MLP(528→384) | Rot-aug + MLP per band | `(T, N, 2, 384)` | Shared within band |
| Flatten | — | Concat bands | `(T, N, 768)` | 768 = 2 × 384 |
| **Input projection** | **Identity** | **No-op (sizes match)** | `(T, N, 768)` | **No CNN stage** |
| **BiGRU Block 1** | **GRU(768→384×2) + LN** | **Recurrent + residual** | `(T, N, 768)` | **T unchanged** |
| **BiGRU Block 2** | **GRU(768→384×2) + LN** | **Recurrent + residual** | `(T, N, 768)` | |
| **BiGRU Block 3** | **GRU(768→384×2) + LN** | **Recurrent + residual** | `(T, N, 768)` | |
| **BiGRU Block 4** | **GRU(768→384×2) + LN** | **Recurrent + residual** | `(T, N, 768)` | No dropout |
| out_projection | Linear(768→768) | Re-scale encoder output | `(T, N, 768)` | |
| out_norm | LayerNorm(768) | Feature normalization | `(T, N, 768)` | |
| Linear head | Linear(768→C) | Character logits | `(T, N, num_classes)` | |
| LogSoftmax | — | Log-probabilities | `(T, N, num_classes)` | |
| CTC Loss | — | Alignment-free seq loss | scalar | Blank = null_class |

Rows in **bold** highlight the components that replace the CNN encoder of the hybrid models.

---

## 9. Configuration & Hyperparameters

All hyperparameters for reproduction are in `config/model/pure_rnn_ctc.yaml`:

```yaml
module:
  _target_: emg2qwerty.pure_rnn.PureRNNCTCModule
  in_features: 528       # = 16 electrodes × 33 freq bins (n_fft//2+1)
  mlp_features: [384]    # MLP hidden/output size; num_features = 2 × 384 = 768
  rnn_hidden_size: 384   # GRU hidden units per direction; output = 768 = num_features
  rnn_num_layers: 4      # Deeper stack (no CNN; 4 layers ≈ comparable capacity)
  rnn_dropout: 0.2       # Inter-layer dropout probability

datamodule:
  window_length: 8000    # 4-second windows at 2 kHz
  padding: [1800, 200]   # 900 ms past context + 100 ms future context
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

| Component | Parameters | vs. CNN/RNN GRU |
| :--- | :--- | :--- |
| SpectrogramNorm (BN2d) | 64 | = same |
| RotationInvariantMLP (×2 bands) | 407,040 | = same |
| Input Projection | 0 (Identity) | — (replaces TDS CNN) |
| **ResidualBiGRUBlock × 4** | **~4 × 1,773,312 ≈ 7.1M** | **+TDS param. replaced** |
| out_projection (Linear 768→768) | 590,592 | = same |
| out_norm (LayerNorm 768) | 1,536 | ≈ same (LN vs LN) |
| Final Linear (768→num_classes) | 768 × num_classes | = same |
| **Total** | **~8.1–8.2 M** (varies by num_classes) | |

**Comparison:**

| Model | Total Params | CNN stages | RNN layers |
| :--- | :--- | :--- | :--- |
| TDS-Conv (baseline) | ~5.5M | 4 TDS blocks | 0 |
| CNN/RNN GRU | ~11–12M | 4 TDS blocks | 2 BiGRU |
| CNN/RNN LSTM | ~13–14M | 4 TDS blocks | 2 BiLSTM |
| **Pure RNN (this model)** | **~8.1M** | **0** | **4 BiGRU** |

The pure RNN is significantly leaner than either hybrid model (~30% fewer parameters than the GRU hybrid) while using more recurrent layers.

**Why fewer parameters despite more GRU layers?**

The TDS fully-connected blocks in the hybrid CNN are the dominant parameter cost:
- 4 TDS FC blocks: `4 × (768×768 + 768 + 768×768 + 768)` ≈ **4.7M parameters**

Removing these 4 large FC blocks saves more parameters than adding 2 extra BiGRU layers costs.

---

## 11. Pure RNN vs. CNN/RNN Hybrid: Design Trade-offs

| Property | Pure RNN (this model) | CNN/RNN Hybrid |
| :--- | :--- | :--- |
| CNN backbone | None | TDS Conv (4 blocks) |
| Temporal shrinkage | 0 frames (T preserved) | 124 frames (497→373) |
| CTC output frames | ~497 | ~373 |
| RNN depth | 4 BiGRU layers | 2 BiGRU layers |
| Residual connections | Per GRU layer | Only within TDS blocks |
| Inter-layer dropout | Yes (between GRU layers) | No (within CNN blocks) |
| Local receptive field | Implicit (via GRU gates) | Explicit (32-frame conv window) |
| Total parameters | ~8.1M | ~11–12M |
| Training speed | Slower per epoch (longer T, deeper RNN) | Faster (T compressed by CNN) |
| Inductive bias | Sequential + gated memory | Local convolution + sequential |
| Best for | Long-range dependencies, no locality assumption | Datasets where local spike shape matters |

**When to prefer the Pure RNN:**
- When the dataset is large enough to learn local patterns from data (no need for CNN's locality prior).
- When the full T ≈ 497 output sequence benefits decoding (more alignment positions for CTC).
- When interpretability of temporal dynamics is desired (GRU states are explicitly sequential).
- As an ablation to quantify how much the CNN contributes vs. the RNN.

---

## 12. How to Reproduce / Instantiate

**Minimal Python instantiation:**

```python
from emg2qwerty.pure_rnn import PureRNNCTCModule

model = PureRNNCTCModule(
    in_features=528,         # 16 * (64//2+1)
    mlp_features=[384],
    rnn_hidden_size=384,
    rnn_num_layers=4,
    rnn_dropout=0.2,
    optimizer={"_target_": "torch.optim.Adam", "lr": 1e-3},
    lr_scheduler={"_target_": "torch.optim.lr_scheduler.CosineAnnealingLR"},
    decoder={"_target_": "emg2qwerty.decoder.GreedyCTCDecoder"},
)
```

**Via Hydra (standard training):**
```bash
python -m emg2qwerty.train model=pure_rnn_ctc
```

**Via the experiment runner:**
```python
# In run_experiments.py EXPERIMENTS list:
dict(
    name          = "Pure-RNN (deep BiGRU) | log_spectrogram",
    model         = "pure_rnn_ctc",
    transforms    = "log_spectrogram",
    rnn_num_layers = 4,
    rnn_hidden_size = 384,
    rnn_dropout   = 0.2,
)
```

**Input tensor format for forward pass:**
```python
# x: (T, N, num_bands=2, electrode_channels=16, freq_bins=33)
x = torch.randn(497, 8, 2, 16, 33)
log_probs = model(x)  # → (T_out ≈ 497, 8, num_classes)
```

---

## 13. Key Design Choices Summary

| Choice | Rationale |
| :--- | :--- |
| No CNN backbone | The CNN's local receptive field is an inductive bias; removing it forces the model to learn temporal structure purely from data via gated recurrence |
| Log-spectrogram (not raw signal) | Reduces sequence length 16× (2kHz→125Hz); frequency content captures motor unit recruitment patterns shared with hybrid models |
| BatchNorm2d per electrode | Normalizes amplitude variation across users and sessions; per-channel independence is critical for multi-user generalization |
| Rotation-invariant MLP | Electrode cuff placement uncertainty modelled by pooling over ±1 electrode offsets; shared with hybrid models |
| Separate MLPs per band | Left and right arm muscles differ; independent weights capture arm-specific biomechanics |
| 4 BiGRU layers (vs 2 in hybrid) | Deeper stack compensates for absent CNN's hierarchical feature extraction |
| Residual connections per layer | Stable gradient flow through 4 stacked recurrent layers; prevents degradation |
| Inter-layer dropout (p=0.2) | Regularization across deeper stack; final layer uses no dropout to preserve classifier input |
| No dropout on final GRU block | Prevents zeroing features that directly feed the output projection and classifier |
| Full T preserved (no CNN shrinkage) | ~33% more CTC output frames; easier alignment for long target sequences |
| GRU over LSTM | ~25% fewer parameters per layer; comparable performance on this dataset; compatible with deeper stacking |
| Bidirectional (both directions) | Offline decoding; anticipatory EMG signals and follow-through patterns require future context; consistent with hybrid models |
| hidden_size=384, output=768=num_features | Eliminates need for residual projections; direct addition in all 4 blocks |
| Linear projection + LayerNorm (output) | Stable GRU output re-scaling; LayerNorm preferred over BatchNorm for variable-length recurrent outputs |
| CTC loss | No frame-level labels available; CTC handles automatic soft alignment during training |
