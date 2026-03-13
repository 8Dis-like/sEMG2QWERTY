import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def check_run(log_dir):
    for root, dirs, files in os.walk(log_dir):
        if any(f for f in files if "events.out.tfevents" in f):
            ea = EventAccumulator(root)
            ea.Reload()
            tags = ea.Tags().get('scalars', [])
            if 'val/CER' in tags:
                events = ea.Scalars('val/CER')
                vals = [e.value for e in events]
                steps = [e.step for e in events]
                print(f"{log_dir}: steps {steps[0]}→{steps[-1]}, CER {max(vals):.1f}→{min(vals):.1f}, rows={len(vals)}")

check_run("outputs/big_conformer_plus/lightning_logs/version_1")
# check_run("logs/2026-03-10/21-18-51")
# check_run("logs/2026-03-10/23-08-37")

def export_run(log_dir, output_file):
    all_data_frames = []
    
    for root, dirs, files in os.walk(log_dir):
        if any(f for f in files if "events.out.tfevents" in f):
            print(f"Extracting from: {root}")
            ea = EventAccumulator(root)
            ea.Reload()
            
            tags = ea.Tags().get('scalars', [])
            for tag in tags:
                events = ea.Scalars(tag)
                df = pd.DataFrame({
                    'step': [e.step for e in events],
                    'value': [e.value for e in events],
                    'metric': tag,
                })
                all_data_frames.append(df)

    master_df = pd.concat(all_data_frames, ignore_index=True)
    # Pivot to wide format — no groupby so no collisions
    wide = master_df.pivot_table(index='step', columns='metric', values='value', aggfunc='last')
    wide.sort_index(inplace=True)
    wide.to_csv(output_file)
    print(f"Saved to {output_file}")

export_run("outputs/big_conformer_plus/lightning_logs/version_1", "run1_big_conf_plus.csv")
# export_run("logs/2026-03-10/21-18-51", "run2_conf_noaug.csv")
# export_run("logs/2026-03-10/23-08-37", "run3_conf_noaug.csv")


for f in ['run1_big_conf_plus.csv']:
    df = pd.read_csv(f)
    val = df[df['val/CER'].notna()]
    print(f"{f}: epochs {val['epoch'].min():.0f}→{val['epoch'].max():.0f}, CER {val['val/CER'].max():.1f}→{val['val/CER'].min():.1f}")




