# Usage Guide

All commands assume you are using the `.venv` created earlier.

---

## Quick Start: Experiment Orchestrator (`run_experiments.py`)

For running multiple models and getting a side-by-side comparison report, use the orchestrator instead of calling `emg2qwerty.train` manually.

### What it does

`run_experiments.py` trains every experiment in its `EXPERIMENTS` list **sequentially**, saves training curves and a JSON summary to `logs/`, and when all runs finish writes a cross-experiment comparison report to `results/`. You only need to edit the top section of the file — no need to touch the rest.

### Running the suite

```bash
python run_experiments.py
```

That's it. Hardware is auto-detected (GPU if available, otherwise CPU).

### Output layout

```
logs/YYYY-MM-DD/HH-MM-SS/         ← one folder per experiment run
  training_curves.png              ← 2×7 dashboard: loss, CER, IER, DER, SER, LR
  curves/                          ← individual PNGs per metric (linear + log scale)
  experiment_summary.json          ← best val CER, test CER/IER/DER/SER, hyperparams
  checkpoints/                     ← best.ckpt and last.ckpt
  lightning_logs/                  ← raw TensorBoard event files

results/YYYY-MM-DD/HH-MM-SS/      ← one folder per full suite run (never overwritten)
  val_cer_comparison.png           ← all experiments' val CER on one plot
  train_loss_comparison.png
  test_cer_bar_chart.png           ← ranked bar chart of final test CER
  experiments.csv                  ← full comparison table (copy into a spreadsheet)
  model_configs.json               ← full config snapshot for every experiment
  best_model.txt                   ← winner announcement + full ranking
```

### Re-generating the report without re-training

If you already have runs in `logs/` and just want to rebuild the comparison plots:

```bash
python -m scripts.report
```

---

### Configuring experiments

Open `run_experiments.py` and edit **two sections** at the top.

#### 1 — `GLOBAL`: defaults applied to every experiment

```python
GLOBAL: dict = dict(
    user         = "single_user",   # dataset split (single_user or generic)
    batch_size   = 32,
    accelerator  = _ACCELERATOR,    # auto-detected; override with "cpu" or "gpu"
    devices      = _DEVICES,        # auto-detected; override with an integer
    lr_scheduler = "linear_warmup_cosine_annealing",
    # max_epochs = 150,             # uncomment to override the default
    # seed       = 1501,
)
```

#### 2 — `EXPERIMENTS`: the list of runs

Each entry is a `dict`. Per-experiment keys **override** the GLOBAL defaults for that run only.

```python
EXPERIMENTS: list[dict] = [
    dict(
        name       = "My experiment name",   # label used in all plots and reports
        model      = "cnn_rnn_ctc",          # which model config to use
        transforms = "log_spectrogram",      # which transform config to use
        # ... optional overrides below ...
    ),
]
```

**Required keys:**

| Key | Description |
| :--- | :--- |
| `name` | Human-readable label shown in all plots and the summary report |
| `model` | Model config name (see choices below) |
| `transforms` | Transform config name (see choices below) |

**Optional override keys:**

| Key | Applies to | Description |
| :--- | :--- | :--- |
| `rnn_num_layers` | GRU, LSTM, Pure-RNN | Number of stacked RNN layers |
| `rnn_hidden_size` | GRU, LSTM, Pure-RNN | Hidden units per direction |
| `rnn_bidirectional` | GRU, LSTM | `True` / `False` |
| `rnn_dropout` | Pure-RNN | Inter-layer dropout probability |
| `lr_scheduler` | all | Override the GLOBAL scheduler for this run |
| `max_epochs` | all | Override training duration for this run |
| `batch_size` | all | Override the GLOBAL batch size for this run |
| `seed` | all | Override the GLOBAL random seed for this run |

**`model` choices:**

| Value | Architecture | Source file |
| :--- | :--- | :--- |
| `cnn_rnn_ctc` | CNN + Bi-directional GRU | `emg2qwerty/cnn_rnn_hybrid.py` |
| `cnn_lstm_ctc` | CNN + Bi-directional LSTM | `emg2qwerty/cnn_lstm_model.py` |
| `pure_rnn_ctc` | Deep Bi-directional GRU (no CNN) | `emg2qwerty/pure_rnn.py` |
| `tds_conv_ctc` | CNN only (baseline) | `emg2qwerty/lightning.py` |

**`transforms` choices:**

| Value | Description |
| :--- | :--- |
| `log_spectrogram` | Log-magnitude STFT spectrogram |
| `log_spectrogram_plus` | Same + SpecAugment and band-rotation during training |

**`lr_scheduler` choices:**

| Value |
| :--- |
| `linear_warmup_cosine_annealing` *(default)* |
| `cosine_annealing` |
| `cosine_annealing_warm_restarts` |
| `reduce_on_plateau` |
| `step` |

---

### Example: adding a new experiment

```python
# In the EXPERIMENTS list, append a new dict:
dict(
    name           = "Pure-RNN | deeper | log_spectrogram_plus",
    model          = "pure_rnn_ctc",
    transforms     = "log_spectrogram_plus",
    rnn_num_layers = 6,
    rnn_hidden_size = 512,
    rnn_dropout    = 0.3,
    max_epochs     = 200,
),
```

To run **only** this experiment (skip the others), comment out or delete the other entries from the list before running.

---

## Manual Training (`emg2qwerty.train`)

Use this when you want a single run with full Hydra override flexibility, or when you don't need the automatic report generation.

### Standard Training

---

## 1. Core Training Commands

The hybrid model is selected using the `model=cnn_rnn_ctc` override.

### Standard Training
Trains the hybrid model on single user #89335547 using the default hyperparameters (2-layer Bi-GRU, 384 hidden units).
```bash
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user
```

### Tuning Hybrid Hyperparameters
You can override RNN-specific settings directly from the command line:
```bash
# Example: Using 3 layers and a larger hidden size
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user \
  module.rnn_num_layers=3 \
  module.rnn_hidden_size=512 \
  module.rnn_bidirectional=False
```


---

## 2. Verification & Local Testing

Use these commands to verify your setup on a local machine before moving to a cluster.

### Fast Development Run
Runs exactly 1 iteration of training, validation, and testing. Perfect for verifying that your GPU/CUDA setup is working with the hybrid architecture. **Note:** This flag disables saving checkpoints.
```bash
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user ++trainer.fast_dev_run=True
```

### Single Epoch Full Pipeline Test
Runs a full epoch of training, validation, and testing while maintaining all callback behaviors (like saving your `.ckpt` files to disk). Use this if you want to verify that checkpointing works.
```bash
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user ++trainer.max_epochs=1
```

---

## 3. Evaluation & Inference

Once you have a trained checkpoint (`.ckpt`), use these commands to evaluate it.

### Standard Greedy Decoding
```bash
python -m emg2qwerty.train train=False model=cnn_rnn_ctc user=single_user checkpoint=path/to/hybrid_model.ckpt
```

### Beam Search + Language Model
For the best accuracy, use the character-level language model:
```bash
python -m emg2qwerty.train train=False model=cnn_rnn_ctc user=single_user \
  decoder=ctc_beam \
  checkpoint=path/to/hybrid_model.ckpt
```

---

## 4. Hardware & Performance

### Multi-GPU Support
The hybrid model supports distributed training. To use all available GPUs:
```bash
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user trainer.accelerator=gpu trainer.devices=-1
```

### Changing Batch Size
If you encounter "Out of Memory" (OOM) errors on smaller GPUs:
```bash
python -m emg2qwerty.train model=cnn_rnn_ctc user=single_user batch_size=16
```
