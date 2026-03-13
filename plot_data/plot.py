import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_stitch(csv_paths):
    all_train = []
    all_val = []
    current_max_epoch = 0
    
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        
        # Clean data
        val = df[df['val/CER'].notna()].copy()
        train = df[df['train/loss'].notna()].copy()
        
        if i > 0:
            start_epoch_in_csv = val['epoch'].min()
            val['epoch'] = val['epoch'] - start_epoch_in_csv + current_max_epoch + 1
            
            start_train_in_csv = train['epoch'].min()
            train['epoch'] = train['epoch'] - start_train_in_csv + current_max_epoch + 1

        all_val.append(val)
        all_train.append(train)
        
        current_max_epoch = val['epoch'].max()
    
    return pd.concat(all_train).reset_index(drop=True), pd.concat(all_val).reset_index(drop=True)


runs = {
    'Resnet (no aug)': [
        'Resnet_no_aug/run1.csv',      
    ],

}

colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['val/CER', 'val/loss', 'val/IER', 'train/loss']
titles  = ['Validation CER', 'Validation Loss', 'Validation IER', 'Train Loss']

for ax, metric, title in zip(axes.flatten(), metrics, titles):
    for (name, paths), color in zip(runs.items(), colors):
        train_df, val_df = load_and_stitch(paths)
        
        if metric.startswith('val'):
            df = val_df[(val_df[metric].notna()) & (val_df['epoch'] > 10)]
            x, y = df['epoch'].values, df[metric].values
            if metric =='val/CER':
                print(f"Minimum Validation CER: {min(y):.4f}")
            if metric == 'val/loss':
                print(f"Avg val loss last 50 epochs: {np.mean(y[:50]):.4f}, Avg val loss last 20: {np.mean(y[:20]):.4f}")
        if metric == 'train/loss':
            df = train_df[(train_df[metric].notna()) & (train_df['epoch'] > 10)]
        else:
            df = train_df[train_df[metric].notna()]
            x, y = df['epoch'].values, df[metric].values
            print(f"Avg train loss last 50 epochs: {np.mean(y[:50]):.4f}, Avg train loss last 20: {np.mean(y[:20]):.4f}")
            
        
        # Raw (transparent)
        ax.plot(x, y, alpha=0.25, color=color, linewidth=0.8)
        # Smoothed
        smoothed = pd.Series(y).rolling(10, min_periods=1).mean().values
        ax.plot(x, smoothed, color=color, linewidth=2, label=name)
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Saved")