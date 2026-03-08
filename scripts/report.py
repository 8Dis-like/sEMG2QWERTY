"""
scripts/report.py
=================
Cross-experiment comparison report generator.

Reads the ``experiment_summary.json`` saved by runner.py for each completed
run, then produces a timestamped output directory under ``results/``:

  results/YYYY-MM-DD/HH-MM-SS/
    val_cer_comparison.png       — val CER curves for every experiment
    train_loss_comparison.png    — train loss curves for every experiment
    val_ier_comparison.png       — val IER curves for every experiment
    val_der_comparison.png       — val DER curves for every experiment
    val_ser_comparison.png       — val SER curves for every experiment
    lr_comparison.png            — adaptive LR schedule for every experiment
    test_cer_bar_chart.png       — ranked bar chart of test CER
    test_ier_bar_chart.png       — ranked bar chart of test IER
    test_der_bar_chart.png       — ranked bar chart of test DER
    test_ser_bar_chart.png       — ranked bar chart of test SER
    experiments.csv              — full comparison table
    model_configs.json           — full config for every experiment (for recreation)
    model_configs.txt            — human-readable version of model_configs.json
    best_model.txt               — winner announcement + full ranking

Each invocation creates a fresh subdirectory so previous results are never
overwritten.

Can also be called standalone to regenerate all graphs from existing logs:

    python -m scripts.report
"""

from __future__ import annotations

import csv
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from scripts.metrics import read_tb_dir
from scripts.plots  import (
    plot_val_cer_comparison,
    plot_test_cer_bar_chart,
    plot_metric_comparison,
    plot_test_metric_bar_chart,
    plot_training_curves,
    plot_individual_curves,
)


# ── Public entry point ────────────────────────────────────────────────────────

def generate_comparison_report(
    completed:   list[tuple[str, Path]],
    results_dir: Optional[Path] = None,
) -> Path:
    """Build a full comparison report from a list of ``(name, run_dir)`` pairs.

    Generates plots, a CSV table, and a plain-text winner summary, all saved
    inside a timestamped subdirectory of ``results/`` so successive runs never
    overwrite each other.

    Args:
        completed:   List of ``(experiment_name, run_directory)`` tuples.
        results_dir: Override the output directory.  If *None* (default), a
                     new ``results/YYYY-MM-DD/HH-MM-SS/`` folder is created.

    Returns:
        The Path to the directory where results were written.
    """
    if results_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        results_dir = Path("results") / stamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating comparison report -> {results_dir}/\n")

    summaries:    list[dict]                  = []
    runs_metrics: list[tuple[str, RunMetrics]] = []

    for name, run_dir in completed:
        summary = _load_summary(name, run_dir)
        if summary:
            summaries.append(summary)

        metrics = _load_metrics(run_dir)
        if metrics:
            runs_metrics.append((name, metrics))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if runs_metrics:
        # Val CER comparison (all experiments)
        plot_val_cer_comparison(
            runs_metrics,
            output_path=results_dir / "val_cer_comparison.png",
        )
        # Train loss comparison
        plot_metric_comparison(
            runs_metrics,
            epochs_attr = "train_loss_epochs",
            vals_attr   = "train_loss_vals",
            title       = "Train Loss",
            ylabel      = "CTC Loss",
            output_path = results_dir / "train_loss_comparison.png",
            log_floor   = 1e-3,
        )
        # Val IER comparison
        plot_metric_comparison(
            runs_metrics,
            epochs_attr = "val_ier_epochs",
            vals_attr   = "val_ier_vals",
            title       = "Validation IER",
            ylabel      = "IER (%)",
            output_path = results_dir / "val_ier_comparison.png",
            ylim        = (0, 100),
        )
        # Val DER comparison
        plot_metric_comparison(
            runs_metrics,
            epochs_attr = "val_der_epochs",
            vals_attr   = "val_der_vals",
            title       = "Validation DER",
            ylabel      = "DER (%)",
            output_path = results_dir / "val_der_comparison.png",
            ylim        = (0, 100),
        )
        # Val SER comparison
        plot_metric_comparison(
            runs_metrics,
            epochs_attr = "val_ser_epochs",
            vals_attr   = "val_ser_vals",
            title       = "Validation SER",
            ylabel      = "SER (%)",
            output_path = results_dir / "val_ser_comparison.png",
            ylim        = (0, 100),
        )
        # Adaptive learning rate comparison
        plot_metric_comparison(
            runs_metrics,
            epochs_attr = "lr_epochs",
            vals_attr   = "lr_vals",
            title       = "Adaptive Learning Rate",
            ylabel      = "LR",
            output_path = results_dir / "lr_comparison.png",
            log_floor   = 1e-9,
        )

    if summaries:
        plot_test_cer_bar_chart(
            summaries,
            output_path=results_dir / "test_cer_bar_chart.png",
        )
        # Test IER / DER / SER bar charts
        plot_test_metric_bar_chart(
            summaries,
            metric_key   = "test_ier",
            fallback_key = None,
            title        = "Test IER by Experiment   (lower is better)",
            output_path  = results_dir / "test_ier_bar_chart.png",
        )
        plot_test_metric_bar_chart(
            summaries,
            metric_key   = "test_der",
            fallback_key = None,
            title        = "Test DER by Experiment   (lower is better)",
            output_path  = results_dir / "test_der_bar_chart.png",
        )
        plot_test_metric_bar_chart(
            summaries,
            metric_key   = "test_ser",
            fallback_key = None,
            title        = "Test SER by Experiment   (lower is better)",
            output_path  = results_dir / "test_ser_bar_chart.png",
        )
        _save_csv(summaries, results_dir / "experiments.csv")
        _save_model_configs(summaries, results_dir)
        _announce_winner(summaries, results_dir)

    return results_dir


# ── Standalone re-run ─────────────────────────────────────────────────────────

def regenerate_from_logs() -> None:
    """Scan all ``logs/`` directories and regenerate all graphs + the comparison report.

    Regenerates per-run training_curves.png and curves/ individual PNGs for
    every run found under logs/, then rebuilds the cross-experiment comparison
    report in results/.

    Usage::

        python -m scripts.report
    """
    summary_paths = sorted(glob.glob("logs/*/*/experiment_summary.json"))
    if not summary_paths:
        print("No experiment_summary.json files found under logs/.")
        return

    completed: list[tuple[str, Path]] = []
    for sp in summary_paths:
        with open(sp) as f:
            s = json.load(f)
        run_dir = Path(sp).parent
        name    = s.get("name", run_dir.name)
        completed.append((name, run_dir))

        # Per-run graphs
        metrics = _load_metrics(run_dir)
        if metrics is None:
            print(f"  [SKIP] No TensorBoard data for '{name}' — skipping per-run plots.")
            continue

        print(f"\n  Regenerating per-run plots for: {name}")
        plot_training_curves(
            metrics,
            name        = name,
            output_path = run_dir / "training_curves.png",
        )
        plot_individual_curves(
            metrics,
            name       = name,
            curves_dir = run_dir / "curves",
        )

    # Cross-experiment comparison
    generate_comparison_report(completed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_summary(name: str, run_dir: Path) -> Optional[dict]:
    path = run_dir / "experiment_summary.json"
    if not path.exists():
        print(f"  [WARNING] No summary found for '{name}' at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_metrics(run_dir: Path) -> Optional[RunMetrics]:
    candidates = list(run_dir.glob("lightning_logs/version_*"))
    tb_dir = max(candidates, key=lambda p: p.stat().st_mtime, default=None)
    if tb_dir is None:
        return None
    return read_tb_dir(tb_dir)


def _save_csv(summaries: list[dict], path: Path) -> None:
    """Write all experiment summaries as a flat CSV table."""
    if not summaries:
        return

    # Collect every hyperparameter key that appears in any summary
    hp_keys = sorted({k for s in summaries for k in s.get("hyperparameters", {})})

    fieldnames = [
        "name", "model",
        "best_val_cer", "best_val_cer_epoch",
        "test_cer", "test_loss", "test_ier", "test_der", "test_ser",
        "run_id", "timestamp",
    ] + hp_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = {**s, **s.get("hyperparameters", {})}
            writer.writerow(row)

    print(f"  CSV     -> {path}")


def _save_model_configs(summaries: list[dict], results_dir: Path) -> None:
    """Write a machine-readable and a human-readable model config file.

    Outputs
    -------
    model_configs.json
        One entry per experiment with the full config snapshot needed to
        recreate the run exactly (model type, transforms, all
        hyperparameters, hardware settings, seed).

    model_configs.txt
        Pretty-printed version of the same data for quick reference.
    """
    if not summaries:
        return

    # ── JSON ──────────────────────────────────────────────────────────────────
    configs = []
    for s in summaries:
        entry = {
            "name":        s.get("name"),
            "run_id":      s.get("run_id"),
            "timestamp":   s.get("timestamp"),
            "full_config": s.get("full_config", s.get("hyperparameters", {})),
        }
        configs.append(entry)

    json_path = results_dir / "model_configs.json"
    with open(json_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"  Configs -> {json_path}")

    # ── Plain text ─────────────────────────────────────────────────────────────
    sep  = "─" * 60
    wide = "═" * 60
    lines = [wide, "  MODEL CONFIGURATIONS", wide, ""]

    for entry in configs:
        lines += [f"  Experiment : {entry['name']}",
                  f"  Run ID     : {entry['run_id']}",
                  f"  Trained at : {entry['timestamp']}",
                  "  Config:"]
        cfg = entry["full_config"]
        if cfg:
            for k, v in cfg.items():
                lines.append(f"    {k:<26}: {v}")
        else:
            lines.append("    (no config recorded)")
        lines += [sep, ""]

    txt_path = results_dir / "model_configs.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Configs -> {txt_path}")


def _announce_winner(summaries: list[dict], results_dir: Path) -> None:
    """Print a formatted winner summary and save it to ``<results_dir>/best_model.txt``."""

    def score(s: dict) -> float:
        v = s.get("test_cer")
        if v is None:
            v = s.get("best_val_cer")
        return v if v is not None else float("inf")

    ranked = sorted(summaries, key=score)
    if not ranked:
        return

    winner = ranked[0]
    sep    = "=" * 62

    lines = [
        sep,
        "  BEST MODEL",
        sep,
        f"  Name            : {winner['name']}",
        f"  Model type      : {winner['model']}",
        f"  Test CER        : {winner.get('test_cer', 'N/A')}",
        (
            f"  Best val CER    : {winner.get('best_val_cer', 'N/A')}"
            f" @ epoch {winner['best_val_cer_epoch']:.0f}"
            if winner.get("best_val_cer_epoch") is not None
            else f"  Best val CER    : {winner.get('best_val_cer', 'N/A')}"
        ),
        "",
        "  Full ranking (by test CER, or best val CER if test unavailable):",
    ]

    for rank, s in enumerate(ranked, 1):
        cer_val = score(s)
        cer_str = f"{cer_val:.2f}%" if cer_val < float("inf") else "N/A"
        marker  = "  <-- BEST" if rank == 1 else ""
        lines.append(f"  {rank}.  {s['name']:<42}  CER = {cer_str}{marker}")

    lines += [
        "",
        "  Winning hyperparameters:",
    ]
    for k, v in winner.get("hyperparameters", {}).items():
        lines.append(f"    {k}: {v}")

    lines.append(sep)
    report = "\n".join(lines)

    print("\n" + report + "\n")

    out_path = results_dir / "best_model.txt"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"  Winner  -> {out_path}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    regenerate_from_logs()
