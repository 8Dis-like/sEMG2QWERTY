"""
scripts/metrics.py
==================
TensorBoard log reader.

Provides the RunMetrics dataclass and read_tb_dir() for extracting
per-epoch scalar metrics from a training run's TensorBoard event files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    """All scalar metrics from one TensorBoard log directory.

    Train loss/CER are logged per *step* (many points per epoch, smooth curves).
    Val loss/CER and LR are logged once per *epoch*.
    Epoch values are fractional — derived by dividing global step by
    steps_per_epoch, which is estimated from the val/CER event sequence.
    """

    # ── Per-step (high-frequency) train metrics ───────────────────────────────
    train_loss_epochs: list[float] = field(default_factory=list)
    train_loss_vals:   list[float] = field(default_factory=list)

    # ── Per-epoch metrics ─────────────────────────────────────────────────────
    val_loss_epochs:  list[float] = field(default_factory=list)
    val_loss_vals:    list[float] = field(default_factory=list)

    train_cer_epochs: list[float] = field(default_factory=list)
    train_cer_vals:   list[float] = field(default_factory=list)

    val_cer_epochs:   list[float] = field(default_factory=list)
    val_cer_vals:     list[float] = field(default_factory=list)

    # ── IER / DER / SER (per epoch, train + val) ──────────────────────────────
    train_ier_epochs: list[float] = field(default_factory=list)
    train_ier_vals:   list[float] = field(default_factory=list)
    train_der_epochs: list[float] = field(default_factory=list)
    train_der_vals:   list[float] = field(default_factory=list)
    train_ser_epochs: list[float] = field(default_factory=list)
    train_ser_vals:   list[float] = field(default_factory=list)

    val_ier_epochs:   list[float] = field(default_factory=list)
    val_ier_vals:     list[float] = field(default_factory=list)
    val_der_epochs:   list[float] = field(default_factory=list)
    val_der_vals:     list[float] = field(default_factory=list)
    val_ser_epochs:   list[float] = field(default_factory=list)
    val_ser_vals:     list[float] = field(default_factory=list)

    # ── Learning rate (per epoch via LearningRateMonitor) ─────────────────────
    lr_epochs: list[float] = field(default_factory=list)
    lr_vals:   list[float] = field(default_factory=list)

    # ── Summary statistics ────────────────────────────────────────────────────
    best_val_cer:       Optional[float] = None
    best_val_cer_epoch: Optional[float] = None
    test_cer:           Optional[float] = None
    test_loss:          Optional[float] = None
    test_ier:           Optional[float] = None
    test_der:           Optional[float] = None
    test_ser:           Optional[float] = None

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def train_acc_vals(self) -> list[float]:
        """Accuracy = 100 - CER (%), aligned with train_cer_epochs."""
        return [100.0 - v for v in self.train_cer_vals]

    @property
    def val_acc_vals(self) -> list[float]:
        """Accuracy = 100 - CER (%), aligned with val_cer_epochs."""
        return [100.0 - v for v in self.val_cer_vals]


# ── Reader ────────────────────────────────────────────────────────────────────

def read_tb_dir(tb_dir: Path) -> Optional[RunMetrics]:
    """Load scalar metrics from a TensorBoard event directory.

    Args:
        tb_dir: Path to a ``lightning_logs/version_N`` directory.

    Returns:
        Populated RunMetrics, or None if no ``val/CER`` data is found
        (which means training has not produced any validation output yet).
    """
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    tags: set[str] = set(ea.Tags().get("scalars", []))

    def load(tag: str) -> tuple[list[float], list[float]]:
        """Return (steps, values) for a tag, or ([], []) if absent."""
        if tag not in tags:
            return [], []
        events = ea.Scalars(tag)
        return [float(e.step) for e in events], [float(e.value) for e in events]

    # val/CER fires exactly once per epoch — used as the epoch-step reference
    val_cer_steps, val_cer_vals = load("val/CER")
    if not val_cer_steps:
        return None

    n = len(val_cer_steps)
    steps_per_epoch: float = val_cer_steps[-1] / max(1.0, n - 1.0)

    def to_epochs(steps: list[float]) -> list[float]:
        return [s / steps_per_epoch for s in steps]

    # Load all available metrics
    tr_loss_s, tr_loss_v = load("train/loss")
    vl_loss_s, vl_loss_v = load("val/loss")
    tr_cer_s,  tr_cer_v  = load("train/CER")

    tr_ier_s, tr_ier_v = load("train/IER")
    tr_der_s, tr_der_v = load("train/DER")
    tr_ser_s, tr_ser_v = load("train/SER")
    vl_ier_s, vl_ier_v = load("val/IER")
    vl_der_s, vl_der_v = load("val/DER")
    vl_ser_s, vl_ser_v = load("val/SER")

    # LearningRateMonitor logs as "lr-Adam", "lr-Adam/pg1", etc.
    lr_tag = next((t for t in sorted(tags) if t.lower().startswith("lr")), None)
    lr_s, lr_v = load(lr_tag) if lr_tag else ([], [])

    _, test_cer_v  = load("test/CER")
    _, test_loss_v = load("test/loss")
    _, test_ier_v  = load("test/IER")
    _, test_der_v  = load("test/DER")
    _, test_ser_v  = load("test/SER")

    # val/loss may have multiple entries per epoch (one per validation batch,
    # all logged at the same global step).  Average duplicates for a cleaner curve.
    vl_loss_s, vl_loss_v = _mean_per_step(vl_loss_s, vl_loss_v)

    # Best val CER
    best_idx           = int(min(range(n), key=lambda i: val_cer_vals[i]))
    best_val_cer       = val_cer_vals[best_idx]
    best_val_cer_epoch = to_epochs(val_cer_steps)[best_idx]

    return RunMetrics(
        train_loss_epochs = to_epochs(tr_loss_s),
        train_loss_vals   = tr_loss_v,
        val_loss_epochs   = to_epochs(vl_loss_s),
        val_loss_vals     = vl_loss_v,
        train_cer_epochs  = to_epochs(tr_cer_s),
        train_cer_vals    = tr_cer_v,
        val_cer_epochs    = to_epochs(val_cer_steps),
        val_cer_vals      = val_cer_vals,
        train_ier_epochs  = to_epochs(tr_ier_s),
        train_ier_vals    = tr_ier_v,
        train_der_epochs  = to_epochs(tr_der_s),
        train_der_vals    = tr_der_v,
        train_ser_epochs  = to_epochs(tr_ser_s),
        train_ser_vals    = tr_ser_v,
        val_ier_epochs    = to_epochs(vl_ier_s),
        val_ier_vals      = vl_ier_v,
        val_der_epochs    = to_epochs(vl_der_s),
        val_der_vals      = vl_der_v,
        val_ser_epochs    = to_epochs(vl_ser_s),
        val_ser_vals      = vl_ser_v,
        lr_epochs         = to_epochs(lr_s),
        lr_vals           = lr_v,
        best_val_cer       = best_val_cer,
        best_val_cer_epoch = best_val_cer_epoch,
        test_cer           = test_cer_v[-1]  if test_cer_v  else None,
        test_loss          = test_loss_v[-1] if test_loss_v else None,
        test_ier           = test_ier_v[-1]  if test_ier_v  else None,
        test_der           = test_der_v[-1]  if test_der_v  else None,
        test_ser           = test_ser_v[-1]  if test_ser_v  else None,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_per_step(
    steps: list[float],
    vals:  list[float],
) -> tuple[list[float], list[float]]:
    """Average values that share the same global step (e.g. val/loss per batch)."""
    if not steps:
        return [], []

    buckets: dict[float, list[float]] = {}
    order:   list[float] = []

    for s, v in zip(steps, vals):
        if s not in buckets:
            order.append(s)
            buckets[s] = []
        buckets[s].append(v)

    avg_vals = [sum(buckets[s]) / len(buckets[s]) for s in order]
    return order, avg_vals
