# Architecture Overview: CNN/RNN Hybrid Models

This project implements two CNN/RNN hybrid models that extend the baseline TDS-Conv architecture by adding a recurrent layer for long-range temporal dependency modelling in sEMG typing decoding.

## Model Variants

| Model | File | Config | Recurrent Unit | Doc |
| :--- | :--- | :--- | :--- | :--- |
| CNN/RNN (GRU) | `emg2qwerty/cnn_rnn_hybrid.py` | `config/model/cnn_rnn_ctc.yaml` | Bi-directional GRU | [ARCHITECTURE_GRU.md](ARCHITECTURE_GRU.md) |
| CNN/RNN (LSTM) | `emg2qwerty/cnn_lstm_model.py` | `config/model/cnn_lstm_ctc.yaml` | Bi-directional LSTM | [ARCHITECTURE_LSTM.md](ARCHITECTURE_LSTM.md) |
| Baseline (CNN only) | `emg2qwerty/lightning.py` | `config/model/tds_conv_ctc.yaml` | None | — |

Detailed architecture documentation — including full data flow diagrams, per-component design rationale, exact tensor shapes, hyperparameter tables, and reproduction instructions — is in the model-specific files above.

---

## Shared Pipeline Summary

Both hybrid models share an identical frontend and CNN backbone; they differ only in the recurrent unit.

```
Raw EMG (2kHz, 16ch/arm, 2 arms)
        ↓  LogSpectrogram (n_fft=64, hop=16) → 125Hz, 33 freq bins
(T≈497, N, 2, 16, 33)
        ↓  SpectrogramNorm (BatchNorm2d per electrode)
        ↓  MultiBandRotationInvariantMLP (528→384 per band, ±1 rotation pooling)
        ↓  Flatten bands → (T, N, 768)
        ↓  TDSConvEncoder (4 × TDSConv2dBlock + TDSFCBlock, kernel=32)
(T-124, N, 768)  [T_cnn ≈ 373]
        ↓  [GRU] or [LSTM]  (hidden=384, 2 layers, bidirectional)
        ↓  Linear(768→768) + LayerNorm
        ↓  Linear(768→num_classes) + LogSoftmax
        ↓  CTCLoss
```

---

## Quick Reference: Default Hyperparameters

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `in_features` | 528 | 16 electrodes × 33 freq bins |
| `mlp_features` | `[384]` | Single-layer MLP |
| `block_channels` | `[24, 24, 24, 24]` | 4 TDS blocks, 24 channels each |
| `kernel_width` | 32 | Each block shrinks T by 31 |
| `rnn_hidden_size` | 384 | Per direction |
| `rnn_num_layers` | 2 | Stacked layers |
| `rnn_bidirectional` | `true` | Forward + backward |
| `window_length` | 8000 | 4 seconds at 2 kHz |
| `batch_size` | 32 | |
| `max_epochs` | 150 | |
