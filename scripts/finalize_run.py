import argparse
import glob
import os
import json
import shutil
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def finalize(exp_name):
    # ── Find the most recent training run (TensorBoard logs) ──────────────────
    tb_dirs = glob.glob("logs/*/*/lightning_logs/version_*")
    if not tb_dirs:
        print("Error: No TensorBoard logs found.")
        return
    
    tb_dir = max(tb_dirs, key=os.path.getmtime)
    run_dir = Path(tb_dir).parents[1]  # logs/<date>/<time>/
    
    print(f"Finalizing run in: {run_dir}")

    # ── Generate Training Curves ──────────────────────────────────────────────
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    
    tags = ea.Tags()["scalars"]
    if "val/CER" not in tags:
        print(f"Warning: 'val/CER' not found in {tb_dir}. Skipping curve generation.")
    else:
        val_cer_events = ea.Scalars("val/CER")
        steps_per_epoch = val_cer_events[-1].step / max(1, (len(val_cer_events) - 1))

        def load_metric(tag):
            if tag not in tags: return [], []
            events = ea.Scalars(tag)
            epochs = [e.step / steps_per_epoch for e in events]
            return epochs, [e.value for e in events]

        raw = {
            "Loss": {k: load_metric(t) for k, t in [("train", "train/loss"), ("val", "val/loss")]},
            "CER":  {k: load_metric(t) for k, t in [("train", "train/CER"),  ("val", "val/CER")]},
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, name in enumerate(["Loss", "CER"]):
            ax = axes[i]
            for split, (epochs, vals) in raw[name].items():
                if vals:
                    ax.plot(epochs, vals, label=split)
            ax.set_title(name)
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        curve_path = run_dir / "training_curves.png"
        plt.tight_layout()
        plt.savefig(curve_path, dpi=100)
        plt.close()
        print(f"  Saved curves to {curve_path}")

    # ── Extract Hyperparameters and Metrics ───────────────────────────────────
    config_path = run_dir / "hydra_configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def last_val(tag): return ea.Scalars(tag)[-1].value if tag in tags else None
    def min_val(tag): return min(e.value for e in ea.Scalars(tag)) if tag in tags else None

    # Determine model type from _target_
    target = cfg["module"]["_target_"]
    model_type = "LSTM" if "LSTM" in target else "GRU"

    summary = {
        "run_name":        exp_name,
        "run_id":          "/".join(run_dir.parts[-2:]),
        "timestamp":       datetime.now().isoformat(),
        "model_type":      model_type,
        "hyperparameters": {
            "model":             cfg["module"]["_target_"],
            "rnn_num_layers":    cfg["module"].get("rnn_num_layers"),
            "rnn_hidden_size":   cfg["module"].get("rnn_hidden_size"),
            "rnn_bidirectional": cfg["module"].get("rnn_bidirectional"),
            "batch_size":        cfg.get("batch_size"),
            "lr":                cfg["optimizer"].get("lr"),
            "seed":              cfg.get("seed"),
        },
        "best_val_CER":    min_val("val/CER"),
        "test_CER":        last_val("test/CER"),
        "test_loss":       last_val("test/loss"),
    }

    with open(run_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {run_dir / 'experiment_summary.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    finalize(args.name)
