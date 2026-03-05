"""
Quick environment verification script for the CNN/RNN hybrid model.
Run these checks before starting a full training run.
"""

import subprocess
import sys

import torch

MODEL = "cnn_rnn_ctc"
USER  = "single_user"
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
DEVICES     = torch.cuda.device_count() if torch.cuda.is_available() else 1

print(f"Accelerator : {ACCELERATOR}")
print(f"Devices     : {DEVICES}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
print()

TESTS = [
    (
        "Fast Dev Run (1 iteration, no checkpoints)",
        [
            sys.executable, "-m", "emg2qwerty.train",
            f"model={MODEL}", f"user={USER}",
            f"trainer.accelerator={ACCELERATOR}", f"trainer.devices={DEVICES}",
            "++trainer.fast_dev_run=True",
        ],
    ),
    (
        "Single Epoch (full pipeline + checkpoint saving)",
        [
            sys.executable, "-m", "emg2qwerty.train",
            f"model={MODEL}", f"user={USER}",
            f"trainer.accelerator={ACCELERATOR}", f"trainer.devices={DEVICES}",
            "++trainer.max_epochs=1",
        ],
    ),
]

for name, cmd in TESTS:
    print(f"{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[FAILED] {name}")
        sys.exit(result.returncode)
    print(f"\n[PASSED] {name}\n")

print("All test runs passed.")
