"""
scripts/plots.py
================
All matplotlib plotting functions.

  plot_training_curves()      — 2×7 per-experiment training dashboard
                                (Loss | CER | Accuracy | IER | DER | SER | LR)
  plot_individual_curves()    — per-experiment individual PNGs, one per metric,
                                saved to <run_dir>/curves/
  plot_metric_comparison()    — all experiments' curves for one metric on one figure
  plot_val_cer_comparison()   — convenience wrapper: val CER comparison
  plot_test_cer_bar_chart()   — horizontal bar chart ranked by test CER
  plot_test_metric_bar_chart()— horizontal bar chart for any scalar test metric
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
BEST_COLOR   = "#4CAF50"   # Material green (best-epoch marker + best bar)
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
    """Save a 2×7 training dashboard for one experiment.

    Layout
    ------
    Columns : CTC Loss | CER (%) | Accuracy (%) | IER (%) | DER (%) | SER (%) | LR
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
            "Insertion Error Rate", "IER (%)",
            metrics.train_ier_epochs, metrics.train_ier_vals,
            metrics.val_ier_epochs,   metrics.val_ier_vals,
            0, 100, 1e-1,
        ),
        (
            "Deletion Error Rate", "DER (%)",
            metrics.train_der_epochs, metrics.train_der_vals,
            metrics.val_der_epochs,   metrics.val_der_vals,
            0, 100, 1e-1,
        ),
        (
            "Substitution Error Rate", "SER (%)",
            metrics.train_ser_epochs, metrics.train_ser_vals,
            metrics.val_ser_epochs,   metrics.val_ser_vals,
            0, 100, 1e-1,
        ),
        (
            "Learning Rate", "LR",
            metrics.lr_epochs, metrics.lr_vals,
            [], [],
            None, None, 1e-9,
        ),
    ]

    fig, axes = plt.subplots(2, 7, figsize=(40, 9))
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
            ax.set_title(f"{title}{scale_label}", fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)

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
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves  -> {output_path}")


# ── Per-experiment individual metric curves ────────────────────────────────────

def plot_individual_curves(
    metrics:     RunMetrics,
    name:        str,
    curves_dir:  Path,
) -> None:
    """Save one PNG per metric for a single experiment.

    Output (all inside ``curves_dir``):
      loss.png       — train + val CTC Loss
      cer.png        — train + val CER
      ier.png        — train + val IER
      der.png        — train + val DER
      ser.png        — train + val SER
      lr.png         — adaptive learning rate

    Each figure has two panels: linear scale (left) and log scale (right).
    """
    curves_dir.mkdir(parents=True, exist_ok=True)

    # (filename, title, ylabel, tr_e, tr_v, val_e, val_v, ylim, log_floor, mark_best_col)
    specs = [
        (
            "loss", "CTC Loss", "CTC Loss",
            metrics.train_loss_epochs, metrics.train_loss_vals,
            metrics.val_loss_epochs,   metrics.val_loss_vals,
            None, 1e-3, False,
        ),
        (
            "cer", "Character Error Rate", "CER (%)",
            metrics.train_cer_epochs, metrics.train_cer_vals,
            metrics.val_cer_epochs,   metrics.val_cer_vals,
            (0, 100), 1e-1, True,
        ),
        (
            "ier", "Insertion Error Rate", "IER (%)",
            metrics.train_ier_epochs, metrics.train_ier_vals,
            metrics.val_ier_epochs,   metrics.val_ier_vals,
            (0, 100), 1e-1, False,
        ),
        (
            "der", "Deletion Error Rate", "DER (%)",
            metrics.train_der_epochs, metrics.train_der_vals,
            metrics.val_der_epochs,   metrics.val_der_vals,
            (0, 100), 1e-1, False,
        ),
        (
            "ser", "Substitution Error Rate", "SER (%)",
            metrics.train_ser_epochs, metrics.train_ser_vals,
            metrics.val_ser_epochs,   metrics.val_ser_vals,
            (0, 100), 1e-1, False,
        ),
        (
            "lr", "Adaptive Learning Rate", "LR",
            metrics.lr_epochs, metrics.lr_vals,
            [], [],
            None, 1e-9, False,
        ),
    ]

    for fname, title, ylabel, tr_e, tr_v, vl_e, vl_v, ylim, log_floor, mark_best in specs:
        if not tr_v and not vl_v:
            continue   # skip metrics not logged by this run

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{name}  —  {title}", fontsize=12, fontweight="bold")

        for ax, use_log in zip(axes, [False, True]):
            if tr_v:
                ax.plot(tr_e, tr_v, color=TRAIN_COLOR, lw=LW,
                        label="train", alpha=0.80)
            if vl_v:
                ax.plot(vl_e, vl_v, color=VAL_COLOR, lw=LW,
                        label="val", alpha=0.95)

            # Best-epoch marker on CER linear panel
            if mark_best and (not use_log) and metrics.best_val_cer is not None:
                ax.axvline(
                    x=metrics.best_val_cer_epoch,
                    color=BEST_COLOR, lw=1.1, ls="--", alpha=0.85, zorder=3,
                )
                ax.annotate(
                    f"best: {metrics.best_val_cer:.1f}%\n@ ep {metrics.best_val_cer_epoch:.0f}",
                    xy=(metrics.best_val_cer_epoch, metrics.best_val_cer),
                    xytext=(7, -14), textcoords="offset points",
                    fontsize=8, color=BEST_COLOR,
                    arrowprops=dict(arrowstyle="-", color=BEST_COLOR, lw=0.8),
                )

            scale = " (log)" if use_log else ""
            ax.set_title(f"{title}{scale}", fontsize=10)
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
                if ylim is not None:
                    ax.set_ylim(*ylim)

            if tr_v or vl_v:
                ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        out = curves_dir / f"{fname}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Individual -> {out}")


# ── Generic cross-experiment metric comparison ─────────────────────────────────

def plot_metric_comparison(
    runs:        list[tuple[str, RunMetrics]],
    epochs_attr: str,
    vals_attr:   str,
    title:       str,
    ylabel:      str,
    output_path: Path,
    ylim:        Optional[tuple[float, float]] = None,
    log_floor:   float = 1e-6,
) -> None:
    """Overlay one metric over epochs for every experiment on one figure.

    The best experiment (lowest best_val_cer) is drawn thicker.
    Both linear and log-scale panels are included.

    Args:
        epochs_attr: Attribute name on RunMetrics for epoch values (e.g. ``"val_cer_epochs"``).
        vals_attr:   Attribute name on RunMetrics for metric values (e.g. ``"val_cer_vals"``).
        title:       Human-readable metric name for the figure title.
        ylabel:      Y-axis label.
        ylim:        Optional (lo, hi) limits for the linear panel.
        log_floor:   Minimum y for the log panel.
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
        f"{title} — All Experiments",
        fontsize=13, fontweight="bold",
    )

    for ax, use_log in zip(axes, [False, True]):
        any_data = False
        for (name, m), color in zip(runs, colors):
            epochs = getattr(m, epochs_attr, [])
            vals   = getattr(m, vals_attr,   [])
            if not vals:
                continue
            any_data = True
            is_best = (name == best_name)
            ax.plot(
                epochs, vals,
                color=color,
                lw=2.2 if is_best else 1.2,
                alpha=1.0 if is_best else 0.65,
                label=f"{name}  [BEST]" if is_best else name,
                zorder=3 if is_best else 2,
            )

        scale = " (log)" if use_log else ""
        ax.set_title(f"{title}{scale}", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)

        if use_log:
            ax.set_yscale("log")
            if any_data:
                ax.set_ylim(bottom=log_floor)
        else:
            if ylim is not None:
                ax.set_ylim(*ylim)

        if any_data:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {title} comparison -> {output_path}")


# ── Cross-experiment val CER comparison ───────────────────────────────────────

def plot_val_cer_comparison(
    runs:        list[tuple[str, RunMetrics]],
    output_path: Path,
) -> None:
    """Overlay val CER over epochs for every experiment on one figure.

    Convenience wrapper around :func:`plot_metric_comparison`.
    """
    plot_metric_comparison(
        runs,
        epochs_attr = "val_cer_epochs",
        vals_attr   = "val_cer_vals",
        title       = "Validation CER",
        ylabel      = "CER (%)",
        output_path = output_path,
        ylim        = (0, 100),
        log_floor   = 1e-1,
    )


# ── Test metric bar charts ─────────────────────────────────────────────────────

def plot_test_metric_bar_chart(
    summaries:   list[dict],
    metric_key:  str,
    fallback_key: Optional[str],
    title:       str,
    output_path: Path,
) -> None:
    """Horizontal bar chart of a scalar test metric per experiment, sorted best-to-worst.

    Args:
        metric_key:   Key to read from each summary dict (e.g. ``"test_cer"``).
        fallback_key: Key to use when metric_key is absent (e.g. ``"best_val_cer"``).
        title:        Chart title.
    """
    if not summaries:
        return

    def _score(s: dict) -> float:
        v = s.get(metric_key)
        if v is None and fallback_key:
            v = s.get(fallback_key)
        return v if v is not None else float("inf")

    sorted_s = sorted(summaries, key=_score)
    names    = [s["name"] for s in sorted_s]
    values   = [_score(s) for s in sorted_s]
    colors   = [BEST_COLOR if i == 0 else BAR_OTHER for i in range(len(names))]

    finite = [v for v in values if v < float("inf")]
    if not finite:
        return

    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.85 + 1.5)))

    bars = ax.barh(names, values, color=colors, edgecolor="white",
                   height=0.55, zorder=3)

    for bar, val in zip(bars, values):
        if val < float("inf"):
            ax.text(
                bar.get_width() + max(finite) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%",
                va="center", ha="left", fontsize=9,
            )

    if values[0] < float("inf"):
        ax.text(
            values[0] / 2, 0,
            "BEST",
            va="center", ha="center",
            fontsize=9, color="white", fontweight="bold", zorder=4,
        )
        ax.axvline(x=values[0], color=BEST_COLOR, lw=1.0, ls="--",
                   alpha=0.7, zorder=2)

    ax.invert_yaxis()
    ax.set_xlim(0, max(finite) * 1.20)
    ax.set_xlabel("(%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart      -> {output_path}")


def plot_test_cer_bar_chart(
    summaries:   list[dict],
    output_path: Path,
) -> None:
    """Horizontal bar chart of test CER per experiment, sorted best-to-worst.

    Falls back to ``best_val_cer`` when ``test_cer`` is unavailable.
    """
    plot_test_metric_bar_chart(
        summaries,
        metric_key   = "test_cer",
        fallback_key = "best_val_cer",
        title        = "Test CER by Experiment   (lower is better)",
        output_path  = output_path,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _qualitative_colors(n: int) -> list:
    """Return n visually distinct colours from matplotlib's tab10 palette."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]
