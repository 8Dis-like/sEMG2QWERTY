"""
scripts/runner.py
=================
Single-experiment execution and post-processing.

Public API
----------
  run_experiment(config)  — train one experiment, save curves + summary,
                            return the run directory on success or None on failure.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.metrics import RunMetrics, read_tb_dir
from scripts.plots  import plot_training_curves


# ── Public entry point ────────────────────────────────────────────────────────

def run_experiment(config: dict) -> Optional[Path]:
    """Train one experiment and save training curves + JSON summary.

    Steps
    -----
    1. Build the Hydra CLI command from *config*.
    2. Run ``python -m emg2qwerty.train`` as a subprocess.
    3. Locate the run directory that was just created under ``logs/``.
    4. Read TensorBoard logs, plot training curves, write ``experiment_summary.json``.

    Args:
        config: Merged dict of GLOBAL defaults and per-experiment overrides
                (produced by ``run_experiments.py``).

    Returns:
        Path to the run directory, or None if training failed.
    """
    cmd = _build_command(config)
    print("  Command: " + " ".join(str(c) for c in cmd[2:]))  # skip python -m prefix

    stamp  = time.time()
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"  Training exited with code {result.returncode}.")
        return None

    run_dir = _find_run_dir(created_after=stamp)
    if run_dir is None:
        print("  Could not locate run directory after training.")
        return None

    tb_dir = _find_tb_dir(run_dir)
    if tb_dir is None:
        print(f"  No TensorBoard event files found in {run_dir}.")
        return run_dir

    metrics = read_tb_dir(tb_dir)
    if metrics is None:
        print("  No val/CER data in TensorBoard logs — skipping post-processing.")
        return run_dir

    plot_training_curves(
        metrics,
        name        = config["name"],
        output_path = run_dir / "training_curves.png",
    )

    _save_summary(config, metrics, run_dir, run_dir / "experiment_summary.json")

    return run_dir


# ── Command builder ───────────────────────────────────────────────────────────

def _build_command(config: dict) -> list[str]:
    """Translate a config dict into a Hydra CLI argument list."""

    cmd = [
        sys.executable, "-m", "emg2qwerty.train",
        f"model={config['model']}",
        f"user={config['user']}",
        f"transforms={config['transforms']}",
        f"trainer.accelerator={config['accelerator']}",
        f"trainer.devices={config['devices']}",
        f"batch_size={config['batch_size']}",
    ]

    # RNN-specific overrides — only meaningful for hybrid (GRU / LSTM) models
    for key in ("rnn_num_layers", "rnn_hidden_size", "rnn_bidirectional"):
        if key in config:
            cmd.append(f"module.{key}={config[key]}")

    # Optional global overrides
    if "lr_scheduler" in config:
        cmd.append(f"lr_scheduler={config['lr_scheduler']}")
    if "max_epochs" in config:
        cmd.append(f"trainer.max_epochs={config['max_epochs']}")
    if "seed" in config:
        cmd.append(f"seed={config['seed']}")

    return cmd


# ── Run-directory locator ─────────────────────────────────────────────────────

def _find_run_dir(created_after: float) -> Optional[Path]:
    """Return the most recently modified ``logs/YYYY-MM-DD/HH-MM-SS`` directory
    that was created at or after *created_after* (Unix timestamp).

    A 5-second buffer is applied to tolerate minor filesystem timing differences.
    """
    candidates = [
        Path(p)
        for p in glob.glob("logs/*/*")
        if os.path.getmtime(p) >= created_after - 5
    ]
    return max(candidates, key=os.path.getmtime, default=None)


def _find_tb_dir(run_dir: Path) -> Optional[Path]:
    """Return the TensorBoard ``version_N`` directory inside a run directory."""
    candidates = list(run_dir.glob("lightning_logs/version_*"))
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


# ── Summary writer ────────────────────────────────────────────────────────────

def _save_summary(
    config:      dict,
    metrics:     RunMetrics,
    run_dir:     Path,
    output_path: Path,
) -> None:
    """Write ``experiment_summary.json`` to the run directory."""

    # Collect only the hyperparameter keys that are relevant
    hp_keys = (
        "transforms", "batch_size", "lr_scheduler",
        "rnn_num_layers", "rnn_hidden_size", "rnn_bidirectional",
        "max_epochs", "seed",
    )

    summary = {
        "name":      config["name"],
        "run_id":    "/".join(run_dir.parts[-2:]),
        "timestamp": datetime.now().isoformat(),
        "model":     config["model"],
        "hyperparameters": {
            k: config[k] for k in hp_keys if k in config
        },
        "best_val_cer":       metrics.best_val_cer,
        "best_val_cer_epoch": metrics.best_val_cer_epoch,
        "test_cer":           metrics.test_cer,
        "test_loss":          metrics.test_loss,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary -> {output_path}")
