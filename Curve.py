"""
plot_training_curves.py
从 training_log.csv 读取日志，绘制 Loss / F1 / Accuracy / LR 曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 读取数据 ============
df = pd.read_csv('training_log.csv')
best_rows = df[df['best'] == 'best']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- 1. Loss ----
ax = axes[0, 0]
ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', lw=1.5)
ax.plot(df['epoch'], df['val_loss'],   label='Val Loss',   color='#ff7f0e', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Loss vs Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 2. F1 Score ----
ax = axes[0, 1]
ax.plot(df['epoch'], df['train_f1'], label='Train F1', color='#1f77b4', lw=1.5)
ax.plot(df['epoch'], df['val_f1'],   label='Val F1',   color='#ff7f0e', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score'); ax.set_title('F1 Score vs Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 3. Accuracy ----
ax = axes[1, 0]
ax.plot(df['epoch'], df['train_acc'], label='Train Acc', color='#1f77b4', lw=1.5)
ax.plot(df['epoch'], df['val_acc'],   label='Val Acc',   color='#ff7f0e', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy vs Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

# ---- 4. Learning Rate ----
ax = axes[1, 1]
ax.plot(df['epoch'], df['lr'], color='#2ca02c', lw=1.5)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('Learning Rate vs Epoch')
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("训练曲线已保存: training_curves.png")
