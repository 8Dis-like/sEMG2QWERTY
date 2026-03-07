"""
scripts/report.py
=================
Cross-experiment comparison report generator.

Reads the ``experiment_summary.json`` saved by runner.py for each completed
run, then produces:

  results/
    val_cer_comparison.png   — val CER curves for every experiment on one plot
    test_cer_bar_chart.png   — ranked horizontal bar chart of test CER
    experiments.csv          — full comparison table
    best_model.txt           — winner announcement + full ranking

Can also be called standalone to regenerate the report from existing logs:

    python -m scripts.report
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Optional

from scripts.metrics import RunMetrics, read_tb_dir
from scripts.plots  import plot_val_cer_comparison, plot_test_cer_bar_chart


RESULTS_DIR = Path("results")


# ── Public entry point ────────────────────────────────────────────────────────

def generate_comparison_report(
    completed: list[tuple[str, Path]],
) -> None:
    """Build a full comparison report from a list of ``(name, run_dir)`` pairs.

    Generates plots, a CSV table, and a plain-text winner summary.

    Args:
        completed: List of ``(experiment_name, run_directory)`` tuples,
                   as returned by ``run_experiments.py``.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"\nGenerating comparison report -> {RESULTS_DIR}/\n")

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
        plot_val_cer_comparison(
            runs_metrics,
            output_path=RESULTS_DIR / "val_cer_comparison.png",
        )

    if summaries:
        plot_test_cer_bar_chart(
            summaries,
            output_path=RESULTS_DIR / "test_cer_bar_chart.png",
        )
        _save_csv(summaries, RESULTS_DIR / "experiments.csv")
        _announce_winner(summaries)


# ── Standalone re-run ─────────────────────────────────────────────────────────

def regenerate_from_logs() -> None:
    """Scan all ``logs/`` directories and regenerate the comparison report.

    Useful when you want to rebuild the report without re-running training.
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
        completed.append((s.get("name", run_dir.name), run_dir))

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
        "test_cer", "test_loss",
        "run_id", "timestamp",
    ] + hp_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = {**s, **s.get("hyperparameters", {})}
            writer.writerow(row)

    print(f"  CSV     -> {path}")


def _announce_winner(summaries: list[dict]) -> None:
    """Print a formatted winner summary and save it to ``results/best_model.txt``."""

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
        f"  Best val CER    : {winner.get('best_val_cer', 'N/A')} "
        f"@ epoch {winner.get('best_val_cer_epoch', '?'):.0f}"
        if winner.get("best_val_cer_epoch") is not None
        else f"  Best val CER    : {winner.get('best_val_cer', 'N/A')}",
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

    out_path = RESULTS_DIR / "best_model.txt"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"  Winner  -> {out_path}")


# ── Standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    regenerate_from_logs()
