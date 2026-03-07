"""
scripts/plots.py
================
All matplotlib plotting functions.

  plot_training_curves()      — 2×4 per-experiment training dashboard
  plot_val_cer_comparison()   — all experiments' val CER on one figure
  plot_test_cer_bar_chart()   — horizontal bar chart ranked by test CER
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scripts.metrics import RunMetrics


# ── Colour palette ─────────────────────────────────────────────────────────────

TRAIN_COLOR  = "#2196F3"   # Material blue
VAL_COLOR    = "#FF5722"   # Material deep-orange
BEST_COLOR   = "#4CAF50"   # Material green
BAR_BEST     = "#4CAF50"
BAR_OTHER    = "#90CAF9"   # Light blue


# ── Global style tweaks (applied once on import) ───────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F9",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "font.size":         10,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#CCCCCC",
})

LW = 1.4   # default line width


# ── Per-experiment training dashboard ─────────────────────────────────────────

def plot_training_curves(
    metrics:     RunMetrics,
    name:        str,
    output_path: Path,
) -> None:
    """Save a 2×4 training dashboard for one experiment.

    Layout
    ------
    Columns : CTC Loss | CER (%) | Accuracy (%) | Learning Rate
    Rows    : Linear scale | Log scale

    The best val-CER epoch is marked with a dashed vertical line on the
    CER and Accuracy columns (linear row only).
    """

    # Column definitions:
    # (title, y-label, train_e, train_v, val_e, val_v, lin_ylo, lin_yhi, log_floor)
    col_defs = [
        (
            "CTC Loss", "CTC Loss",
            metrics.train_loss_epochs, metrics.train_loss_vals,
            metrics.val_loss_epochs,   metrics.val_loss_vals,
            0, None, 1e-3,
        ),
        (
            "Character Error Rate", "CER (%)",
            metrics.train_cer_epochs, metrics.train_cer_vals,
            metrics.val_cer_epochs,   metrics.val_cer_vals,
            0, 100, 1e-1,
        ),
        (
            "Accuracy  (1 − CER)", "Accuracy (%)",
            metrics.train_cer_epochs, metrics.train_acc_vals,
            metrics.val_cer_epochs,   metrics.val_acc_vals,
            0, 100, 1e-1,
        ),
        (
            "Learning Rate", "LR",
            metrics.lr_epochs, metrics.lr_vals,
            [], [],
            None, None, 1e-9,
        ),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(
        f"{name}  —  Training Curves",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, (title, ylabel, tr_e, tr_v, vl_e, vl_v, lin_lo, lin_hi, log_floor) in enumerate(col_defs):
        for row, use_log in enumerate([False, True]):
            ax = axes[row][col]

            # ── Draw lines ────────────────────────────────────────────────────
            if tr_v:
                ax.plot(tr_e, tr_v, color=TRAIN_COLOR, lw=LW,
                        label="train", alpha=0.80)
            if vl_v:
                ax.plot(vl_e, vl_v, color=VAL_COLOR, lw=LW,
                        label="val", alpha=0.95)

            # ── Best-epoch marker (CER and Accuracy, linear row only) ─────────
            if (not use_log) and (col in (1, 2)) and (metrics.best_val_cer is not None):
                best_y = (
                    metrics.best_val_cer
                    if col == 1
                    else 100.0 - metrics.best_val_cer
                )
                ax.axvline(
                    x=metrics.best_val_cer_epoch,
                    color=BEST_COLOR, lw=1.1, ls="--", alpha=0.85, zorder=3,
                )
                ax.annotate(
                    f"best: {best_y:.1f}%\n@ ep {metrics.best_val_cer_epoch:.0f}",
                    xy=(metrics.best_val_cer_epoch, best_y),
                    xytext=(7, -14), textcoords="offset points",
                    fontsize=7.5, color=BEST_COLOR,
                    arrowprops=dict(arrowstyle="-", color=BEST_COLOR, lw=0.8),
                )

            # ── Scale & limits ────────────────────────────────────────────────
            scale_label = " (log)" if use_log else ""
            ax.set_title(f"{title}{scale_label}", fontsize=10)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(labelsize=8)

            if use_log:
                all_pos = [v for v in (tr_v + vl_v) if v > 0]
                if all_pos:
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=max(log_floor, min(all_pos) * 0.7))
                    ax.yaxis.set_major_formatter(
                        ticker.LogFormatterSciNotation(labelOnlyBase=False)
                    )
            else:
                if lin_lo is not None or lin_hi is not None:
                    ax.set_ylim(lin_lo, lin_hi)

            if tr_v or vl_v:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves  -> {output_path}")


# ── Cross-experiment val CER comparison ───────────────────────────────────────

def plot_val_cer_comparison(
    runs:        list[tuple[str, RunMetrics]],
    output_path: Path,
) -> None:
    """Overlay val CER over epochs for every experiment on one figure.

    The best experiment (lowest best_val_cer) is drawn thicker and labelled
    ``[BEST]``.  Both linear and log-scale panels are included.
    """
    if not runs:
        return

    def best_cer(r: tuple[str, RunMetrics]) -> float:
        v = r[1].best_val_cer
        return v if v is not None else float("inf")

    best_name = min(runs, key=best_cer)[0]
    colors     = _qualitative_colors(len(runs))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Validation CER — All Experiments",
        fontsize=13, fontweight="bold",
    )

    for ax, use_log in zip(axes, [False, True]):
        for (name, m), color in zip(runs, colors):
            if not m.val_cer_vals:
                continue
            is_best = (name == best_name)
            ax.plot(
                m.val_cer_epochs, m.val_cer_vals,
                color=color,
                lw=2.2 if is_best else 1.2,
                alpha=1.0 if is_best else 0.65,
                label=f"{name}  [BEST]" if is_best else name,
                zorder=3 if is_best else 2,
            )

        scale = " (log)" if use_log else ""
        ax.set_title(f"Val CER{scale}", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("CER (%)", fontsize=10)
        ax.tick_params(labelsize=9)

        if use_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 100)

        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Val CER comparison -> {output_path}")


# ── Test CER bar chart ─────────────────────────────────────────────────────────

def plot_test_cer_bar_chart(
    summaries:   list[dict],
    output_path: Path,
) -> None:
    """Horizontal bar chart of test CER per experiment, sorted best-to-worst.

    The best experiment is coloured green; all others are light blue.
    Falls back to ``best_val_cer`` when ``test_cer`` is unavailable.
    """
    if not summaries:
        return

    def _score(s: dict) -> float:
        v = s.get("test_cer")
        if v is None:
            v = s.get("best_val_cer")
        return v if v is not None else float("inf")

    # Sort ascending so best (lowest CER) ends up at the top after invert_yaxis
    sorted_s = sorted(summaries, key=_score)
    names    = [s["name"] for s in sorted_s]
    values   = [_score(s) for s in sorted_s]
    colors   = [BAR_BEST if i == 0 else BAR_OTHER for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.85 + 1.5)))

    bars = ax.barh(names, values, color=colors, edgecolor="white",
                   height=0.55, zorder=3)

    # Value labels
    for bar, val in zip(bars, values):
        if val < float("inf"):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center", ha="left", fontsize=9,
            )

    # "BEST" label inside the first (best) bar
    if values and values[0] < float("inf"):
        ax.text(
            values[0] / 2, 0,
            "BEST",
            va="center", ha="center",
            fontsize=9, color="white", fontweight="bold", zorder=4,
        )
        # Reference line at best CER
        ax.axvline(x=values[0], color=BAR_BEST, lw=1.0, ls="--",
                   alpha=0.7, zorder=2)

    ax.invert_yaxis()   # best at top
    ax.set_xlim(0, max(v for v in values if v < float("inf")) * 1.20)
    ax.set_xlabel("CER (%)", fontsize=11)
    ax.set_title(
        "Test CER by Experiment   (lower is better)",
        fontsize=12, fontweight="bold",
    )
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CER bar chart  -> {output_path}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _qualitative_colors(n: int) -> list:
    """Return n visually distinct colours from matplotlib's tab10 palette."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]
