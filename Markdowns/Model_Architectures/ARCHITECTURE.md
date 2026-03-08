# Architecture Overview: CNN/RNN Hybrid and Pure RNN Models

This project implements three sequence models for sEMG typing decoding, spanning CNN/RNN hybrids and a purely recurrent approach.

## Model Variants

| Model | File | Config | CNN Backbone | Recurrent Unit | Doc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CNN/RNN (GRU) | `emg2qwerty/cnn_rnn_hybrid.py` | `config/model/cnn_rnn_ctc.yaml` | TDS Conv (4 blocks) | Bi-directional GRU (2 layers) | [ARCHITECTURE_GRU.md](ARCHITECTURE_GRU.md) |
| CNN/RNN (LSTM) | `emg2qwerty/cnn_lstm_model.py` | `config/model/cnn_lstm_ctc.yaml` | TDS Conv (4 blocks) | Bi-directional LSTM (2 layers) | [ARCHITECTURE_LSTM.md](ARCHITECTURE_LSTM.md) |
| **Pure RNN (deep BiGRU)** | **`emg2qwerty/pure_rnn.py`** | **`config/model/pure_rnn_ctc.yaml`** | **None** | **Bi-directional GRU (4 layers, residual)** | **[ARCHITECTURE_PURE_RNN.md](ARCHITECTURE_PURE_RNN.md)** |
| Baseline (CNN only) | `emg2qwerty/lightning.py` | `config/model/tds_conv_ctc.yaml` | TDS Conv (4 blocks) | None | — |

Detailed architecture documentation — including full data flow diagrams, per-component design rationale, exact tensor shapes, hyperparameter tables, and reproduction instructions — is in the model-specific files above.

---

## Shared Pipeline Summary

All four models share an identical input frontend. They differ in the encoder stage that follows.

```
Raw EMG (2kHz, 16ch/arm, 2 arms)
        ↓  LogSpectrogram (n_fft=64, hop=16) → 125Hz, 33 freq bins
(T≈497, N, 2, 16, 33)
        ↓  SpectrogramNorm (BatchNorm2d per electrode)
        ↓  MultiBandRotationInvariantMLP (528→384 per band, ±1 rotation pooling)
        ↓  Flatten bands → (T, N, 768)
        ↓  [ENCODER — see below]
        ↓  Linear(768→num_classes) + LogSoftmax
        ↓  CTCLoss
```

**Encoder variants:**

```
CNN/RNN (GRU / LSTM):
  TDSConvEncoder (4 × TDSConv2dBlock + TDSFCBlock, kernel=32)
  → (T-124, N, 768)  [T_cnn ≈ 373]
  → GRU or LSTM (hidden=384, 2 layers, bidirectional)
  → Linear(768→768) + LayerNorm

Pure RNN (deep BiGRU):
  DeepBiGRUEncoder (4 × ResidualBiGRUBlock, no CNN)
  → (T, N, 768)      [T unchanged ≈ 497]
  → Linear(768→768) + LayerNorm
```

---

## Quick Reference: Default Hyperparameters

### Shared (all models)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `in_features` | 528 | 16 electrodes × 33 freq bins |
| `mlp_features` | `[384]` | Single-layer MLP → `num_features = 768` |
| `window_length` | 8000 | 4 seconds at 2 kHz |
| `batch_size` | 32 | |
| `max_epochs` | 150 | |

### CNN/RNN Hybrids (GRU and LSTM)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `block_channels` | `[24, 24, 24, 24]` | 4 TDS blocks, 24 channels each |
| `kernel_width` | 32 | Each block shrinks T by 31; total shrinkage = 124 |
| `rnn_hidden_size` | 384 | Per direction |
| `rnn_num_layers` | 2 | Stacked layers |
| `rnn_bidirectional` | `true` | Forward + backward |

### Pure RNN (deep BiGRU)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `rnn_hidden_size` | 384 | Per direction; output = 768 = num_features |
| `rnn_num_layers` | 4 | Deeper stack; no CNN to pre-extract features |
| `rnn_dropout` | 0.2 | Inter-layer dropout (no CNN blocks to regularize) |
