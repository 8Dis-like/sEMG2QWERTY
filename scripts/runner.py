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
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.metrics import RunMetrics, read_tb_dir
from scripts.plots  import plot_training_curves, plot_individual_curves


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

    # Snapshot existing log dirs before training so we can identify the new one
    # by set-difference rather than mtime (mtime is unreliable on Google Drive).
    dirs_before = set(glob.glob("logs/*/*"))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"  Training exited with code {result.returncode}.")
        return None

    run_dir = _find_run_dir(dirs_before)
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

    plot_individual_curves(
        metrics,
        name       = config["name"],
        curves_dir = run_dir / "curves",
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

def _find_run_dir(dirs_before: set[str]) -> Optional[Path]:
    """Return the new ``logs/YYYY-MM-DD/HH-MM-SS`` directory created by the
    training run, identified by set-difference from *dirs_before*.

    Falls back to the most recently modified candidate if no new directory is
    found (e.g. if the run reused an existing path).
    """
    dirs_after = set(glob.glob("logs/*/*"))
    new_dirs   = dirs_after - dirs_before

    if not new_dirs:
        print("  [WARNING] No new log directory detected — falling back to most-recently "
              "modified directory. Results may be attributed to the wrong run.")
    candidates = new_dirs if new_dirs else dirs_after
    return max((Path(p) for p in candidates), key=os.path.getmtime, default=None)


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

    # Keys that describe the model architecture / training recipe (used for
    # CSV columns and the winner report).
    hp_keys = (
        "transforms", "batch_size", "lr_scheduler",
        "rnn_num_layers", "rnn_hidden_size", "rnn_bidirectional",
        "max_epochs", "seed",
    )

    # Full config snapshot — every key that has a JSON-serialisable scalar
    # value, so the run can be recreated exactly from this file alone.
    _SKIP = {"name"}   # already stored as a top-level field
    full_config = {
        k: v for k, v in config.items()
        if k not in _SKIP and isinstance(v, (str, int, float, bool, type(None)))
    }

    summary = {
        "name":      config["name"],
        "run_id":    "/".join(run_dir.parts[-2:]),
        "timestamp": datetime.now().isoformat(),
        "model":     config["model"],
        "full_config": full_config,
        "hyperparameters": {
            k: config[k] for k in hp_keys if k in config
        },
        "best_val_cer":       metrics.best_val_cer,
        "best_val_cer_epoch": metrics.best_val_cer_epoch,
        "test_cer":           metrics.test_cer,
        "test_loss":          metrics.test_loss,
        "test_ier":           metrics.test_ier,
        "test_der":           metrics.test_der,
        "test_ser":           metrics.test_ser,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary -> {output_path}")
