#!/usr/bin/env python
"""
run_experiments.py  —  sEMG2QWERTY Experiment Orchestrator
===========================================================
Edit GLOBAL and EXPERIMENTS below, then run:

    python run_experiments.py

Each experiment is trained in sequence.  After every run, training curves
are saved to the run's log directory.  Once all experiments finish, a
cross-experiment comparison report is written to results/.

Output layout
-------------
  logs/YYYY-MM-DD/HH-MM-SS/        ← one directory per experiment run
    training_curves.png             ← 2×7 dashboard (loss, CER, acc, IER, DER, SER, LR)
    curves/                         ← individual PNGs per metric (linear + log)
      loss.png  cer.png  ier.png  der.png  ser.png  lr.png
    experiment_summary.json         ← metrics + hyperparameters for this run
    checkpoints/                    ← best and last checkpoints
    lightning_logs/                 ← raw TensorBoard event files

  results/YYYY-MM-DD/HH-MM-SS/     ← one directory per suite run (never overwritten)
    val_cer_comparison.png          ← all experiments' val CER on one plot
    train_loss_comparison.png       ← all experiments' train loss on one plot
    val_ier_comparison.png          ← all experiments' val IER on one plot
    val_der_comparison.png          ← all experiments' val DER on one plot
    val_ser_comparison.png          ← all experiments' val SER on one plot
    lr_comparison.png               ← all experiments' LR schedule on one plot
    test_cer_bar_chart.png          ← ranked bar chart of test CER
    test_ier_bar_chart.png          ← ranked bar chart of test IER
    test_der_bar_chart.png          ← ranked bar chart of test DER
    test_ser_bar_chart.png          ← ranked bar chart of test SER
    experiments.csv                 ← full comparison table
    model_configs.json              ← full config snapshot for every experiment
    model_configs.txt               ← human-readable version of model_configs.json
    best_model.txt                  ← winner announcement + full ranking

Re-generating only the report (without re-training):

    python -m scripts.report
"""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.runner import run_experiment
from scripts.report import generate_comparison_report


# ── Hardware (auto-detected; override manually if needed) ─────────────────────

_ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
_DEVICES     = torch.cuda.device_count() if torch.cuda.is_available() else 1


# ═════════════════════════════════════════════════════════════════════════════
#  GLOBAL DEFAULTS  —  applied to every experiment
# ═════════════════════════════════════════════════════════════════════════════

GLOBAL: dict = dict(
    user         = "single_user",
    batch_size   = 32,
    accelerator  = _ACCELERATOR,
    devices      = _DEVICES,
    lr_scheduler = "linear_warmup_cosine_annealing",
    # max_epochs = 150,   # uncomment to override the Hydra config default
    # seed       = 1501,
)


# ═════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTS  —  add, remove, or edit dicts freely
# ═════════════════════════════════════════════════════════════════════════════
#
# Required keys : name, model, transforms
# Optional keys : rnn_num_layers, rnn_hidden_size, rnn_bidirectional,
#                 rnn_dropout, lr_scheduler, max_epochs, batch_size, seed
#
# model choices       : cnn_rnn_ctc (GRU), cnn_lstm_ctc (LSTM), tds_conv_ctc,
#                       pure_rnn_ctc (deep BiGRU, no CNN)
# transforms choices  : log_spectrogram, log_spectrogram_plus
# lr_scheduler choices: linear_warmup_cosine_annealing, cosine_annealing,
#                       cosine_annealing_warm_restarts, reduce_on_plateau, step

EXPERIMENTS: list[dict] = [
    dict(
        name              = "CNN/RNN-GRU | log_spectrogram",
        model             = "cnn_rnn_ctc",
        transforms        = "log_spectrogram",
        rnn_num_layers    = 2,
        rnn_hidden_size   = 384,
        rnn_bidirectional = True,
    ),
    dict(
        name              = "CNN/RNN-GRU | log_spectrogram_plus",
        model             = "cnn_rnn_ctc",
        transforms        = "log_spectrogram_plus",
        rnn_num_layers    = 2,
        rnn_hidden_size   = 384,
        rnn_bidirectional = True,
    ),
    dict(
        name              = "CNN/RNN-LSTM | log_spectrogram",
        model             = "cnn_lstm_ctc",
        transforms        = "log_spectrogram",
        rnn_num_layers    = 2,
        rnn_hidden_size   = 384,
        rnn_bidirectional = True,
    ),
    dict(
        name              = "CNN/RNN-LSTM | log_spectrogram_plus",
        model             = "cnn_lstm_ctc",
        transforms        = "log_spectrogram_plus",
        rnn_num_layers    = 2,
        rnn_hidden_size   = 384,
        rnn_bidirectional = True,
    ),
    dict(
        name              = "Pure-RNN (deep BiGRU) | log_spectrogram",
        model             = "pure_rnn_ctc",
        transforms        = "log_spectrogram",
        rnn_num_layers    = 4,
        rnn_hidden_size   = 384,
        rnn_dropout       = 0.2,
    ),
    dict(
        name              = "Pure-RNN (deep BiGRU) | log_spectrogram_plus",
        model             = "pure_rnn_ctc",
        transforms        = "log_spectrogram_plus",
        rnn_num_layers    = 4,
        rnn_hidden_size   = 384,
        rnn_dropout       = 0.2,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN  —  no edits needed below this line
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _print_hardware_info()

    completed: list[tuple[str, Path]] = []
    total = len(EXPERIMENTS)
    sep   = "=" * 68

    for i, exp in enumerate(EXPERIMENTS, 1):
        cfg = {**GLOBAL, **exp}    # per-experiment keys override GLOBAL

        print(f"\n{sep}")
        print(f"  [{i}/{total}]  {cfg['name']}")
        print(sep)

        run_dir = run_experiment(cfg)

        if run_dir is not None:
            completed.append((cfg["name"], run_dir))
            print(f"\n  Saved to: {run_dir}")
        else:
            print(f"\n  FAILED — skipping '{cfg['name']}'")

    print(f"\n{sep}")
    print(f"  Completed {len(completed)} / {total} experiments")
    print(sep)

    if completed:
        generate_comparison_report(completed)


def _print_hardware_info() -> None:
    sep = "=" * 68
    print(sep)
    print("  sEMG2QWERTY — Experiment Runner")
    print(sep)
    print(f"  Accelerator : {_ACCELERATOR}")
    print(f"  Devices     : {_DEVICES}")
    if torch.cuda.is_available():
        for i in range(_DEVICES):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Experiments : {len(EXPERIMENTS)}")
    for e in EXPERIMENTS:
        print(f"    - {e['name']}")
    print(sep)


if __name__ == "__main__":
    main()
