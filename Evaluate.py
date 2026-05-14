"""
evaluate_and_plot.py
加载模型，在验证集上推理，绘制混淆矩阵 / ROC / PR 曲线
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)

from Model import Net
from Dataloader import RobustImageFolder

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ★★★ 需要你修改的参数 ★★★
# ============================================================
MODEL_PATH  = 'best_model.pth'                                    # 模型权重路径
DATA_ROOT   = '../trash_division_data/ultimate_4_class/'          # 数据集根目录
BATCH_SIZE  = 64
IMAGE_SIZE  = 256
NUM_WORKERS = 4
# ============================================================

# ---------- 1. 加载验证集 ----------
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_dataset = RobustImageFolder(
    root=os.path.join(DATA_ROOT, 'val'),
    transform=val_transform,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, drop_last=False)

class_names = val_dataset.classes
num_classes = len(class_names)
print(f"类别: {class_names}")

# ---------- 2. 加载模型 ----------
device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(device)
model = Net(num_classes=num_classes)
state_dict = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
elif 'model' in state_dict:
    state_dict = state_dict['model']
model.load_state_dict(state_dict)
model = model.to(device).eval()
print("模型加载完成")

# ---------- 3. 推理 ----------
all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_probs  = np.concatenate(all_probs)
all_preds  = np.argmax(all_probs, axis=1)
print(f"推理完成, 共 {len(all_labels)} 样本")

# ============================================================
# ① 混淆矩阵
# ============================================================
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap='Blues', values_format='d', xticks_rotation=30)
ax.set_title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("混淆矩阵已保存: confusion_matrix.png")

# ============================================================
# ② ROC 曲线 (One-vs-Rest + Macro-average)
# ============================================================
one_hot = np.eye(num_classes)[all_labels]
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(8, 7))
fpr_d, tpr_d, auc_d = {}, {}, {}

for i in range(num_classes):
    fpr_d[i], tpr_d[i], _ = roc_curve(one_hot[:, i], all_probs[:, i])
    auc_d[i] = auc(fpr_d[i], tpr_d[i])
    ax.plot(fpr_d[i], tpr_d[i], color=colors[i], lw=2,
            label=f'{class_names[i]} (AUC={auc_d[i]:.4f})')

# Macro-average
all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in range(num_classes)]))
mean_tpr = sum(np.interp(all_fpr, fpr_d[i], tpr_d[i]) for i in range(num_classes)) / num_classes
macro_auc = auc(all_fpr, mean_tpr)
ax.plot(all_fpr, mean_tpr, 'navy', lw=2, ls='--',
        label=f'Macro-avg (AUC={macro_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve', fontsize=14)
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("ROC 曲线已保存: roc_curve.png")

# ============================================================
# ③ Precision-Recall 曲线
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

for i in range(num_classes):
    prec, rec, _ = precision_recall_curve(one_hot[:, i], all_probs[:, i])
    ap = average_precision_score(one_hot[:, i], all_probs[:, i])
    ax.plot(rec, prec, color=colors[i], lw=2,
            label=f'{class_names[i]} (AP={ap:.4f})')

ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve', fontsize=14)
ax.legend(loc='best'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("PR 曲线已保存: pr_curve.png")
