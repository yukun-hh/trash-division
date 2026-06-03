"""
baseline/compare_models.py
多模型对比：ROC 曲线 + 准确率柱状图
添加新模型只需在 MODELS 列表加一行，无需修改绘图代码
author: yukun-hh
date: 2026-5-14
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    precision_recall_curve, average_precision_score,
)

from Model import Net
from Dataloader import RobustImageFolder
from baseline.VGG_KNN import VGGKNNBaseline
from baseline.ResNet34_Pretrained_10pct import get_resnet34_10pct_preds
from baseline.HOG_Baseline import get_hog_lr_preds

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ★★★ 可配置参数 ★★★
# ============================================================
DATA_ROOT   = '../../trash_division_data/ultimate_4_class/'
BATCH_SIZE  = 32
IMAGE_SIZE  = 256
NUM_WORKERS = 4
K_KNN       = 5
# ============================================================

CLASS_NAMES = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']
NUM_CLASSES = 4

# ============================================================
# 预测函数 — 每个函数签名: (train_loader, val_loader, device) -> (y_true, y_preds, y_probs)
# ============================================================

def get_resnet34_preds(train_loader, val_loader, device):
    model = Net(num_classes=NUM_CLASSES)
    state_dict = torch.load('../best_model.pth', map_location='cpu')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    y_true, y_preds, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='ResNet-34'):
            images, labels = images.to(device), labels
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.append(labels.numpy())
            y_preds.append(preds.cpu().numpy())
            y_probs.append(probs.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_preds), np.concatenate(y_probs)


def get_vgg_knn_preds(train_loader, val_loader, device):
    baseline = VGGKNNBaseline(k=K_KNN, device=device)
    baseline.fit(train_loader)
    return baseline.predict(val_loader)


# ============================================================
# ★ 模型注册表 — 添加新模型只需在这里加一行 ★
# ============================================================

MODELS = [
    ('ResNet-34',                  get_resnet34_preds),
    ('ResNet-34 (10% Fine-tune)',  get_resnet34_10pct_preds),
    ('VGG16 + KNN  (K=5)',        get_vgg_knn_preds),
    ('HOG + LogisticRegression',   get_hog_lr_preds),
    # 未来轻松扩展示例：
    # ('ResNet-18 (pretrained)', get_resnet18_preds),
    # ('ResNet-50 (pretrained)', get_resnet50_preds),
    # ('ResNet-34 (finetuned)',  get_finetuned_preds),
]

# ============================================================
# 调色板 (扩展时无需修改)
# ============================================================
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def compute_macro_roc(y_true, y_probs):
    one_hot = np.eye(NUM_CLASSES)[y_true]
    fpr_dict, tpr_dict = {}, {}
    for c in range(NUM_CLASSES):
        fpr_dict[c], tpr_dict[c], _ = roc_curve(one_hot[:, c], y_probs[:, c])
    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


def compute_macro_pr(y_true, y_probs):
    one_hot = np.eye(NUM_CLASSES)[y_true]
    prec_dict, rec_dict = {}, {}
    for c in range(NUM_CLASSES):
        prec_dict[c], rec_dict[c], _ = precision_recall_curve(one_hot[:, c], y_probs[:, c])
    all_rec = np.linspace(0, 1, 200)
    mean_prec = np.zeros_like(all_rec)
    for c in range(NUM_CLASSES):
        mean_prec += np.interp(all_rec, rec_dict[c][::-1], prec_dict[c][::-1])
    mean_prec /= NUM_CLASSES
    macro_ap = average_precision_score(one_hot, y_probs, average='macro')
    return all_rec, mean_prec, macro_ap


def sanitize_filename(name):
    return re.sub(r'[^\w\-_]', '_', name).strip('_')


def preds_csv_path(out_dir, model_name):
    safe = sanitize_filename(model_name)
    return os.path.join(out_dir, f'{safe}_preds.csv')


def save_preds_csv(path, y_true, y_preds, y_probs):
    header = 'y_true,y_pred,' + ','.join(f'prob_{c}' for c in range(NUM_CLASSES))
    data = np.column_stack([y_true.astype(float), y_preds.astype(float), y_probs])
    np.savetxt(path, data, delimiter=',', header=header, comments='', fmt='%.6f')


def load_preds_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    y_true = data[:, 0].astype(int)
    y_preds = data[:, 1].astype(int)
    y_probs = data[:, 2:2 + NUM_CLASSES]
    return y_true, y_preds, y_probs


if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RobustImageFolder(root=os.path.join(DATA_ROOT, 'train'),
                                      transform=val_transform)
    val_dataset = RobustImageFolder(root=os.path.join(DATA_ROOT, 'val'),
                                    transform=val_transform)
    print(f"训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    # ———— 评估所有模型（有缓存则跳过）————
    results = {}
    for name, func in MODELS:
        print(f"\n{'='*50}")
        csv_path = preds_csv_path(out_dir, name)
        if os.path.exists(csv_path):
            print(f"加载缓存: {os.path.basename(csv_path)}")
            y_true, y_preds, y_probs = load_preds_csv(csv_path)
        else:
            print(f"评估: {name}")
            y_true, y_preds, y_probs = func(train_loader, val_loader, device)
            save_preds_csv(csv_path, y_true, y_preds, y_probs)
            print(f"  预测数据已保存: {os.path.basename(csv_path)}")
        acc = accuracy_score(y_true, y_preds)
        fpr, tpr, roc_auc = compute_macro_roc(y_true, y_probs)
        rec, prec, macro_ap = compute_macro_pr(y_true, y_probs)
        results[name] = {'y_true': y_true, 'y_preds': y_preds, 'y_probs': y_probs,
                         'acc': acc, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc,
                         'rec': rec, 'prec': prec, 'ap': macro_ap}
        print(f"  Accuracy: {acc:.4f}  |  Macro-AUC: {roc_auc:.4f}  |  Macro-AP: {macro_ap:.4f}")

    # ———— ROC 对比图 ————
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, r) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(r['fpr'], r['tpr'], color=color, lw=2,
                label=f"{name} (AUC={r['auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison (Macro-Average)', fontsize=14)
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, 'roc_comparison.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nROC 对比图已保存: {roc_path}")

    # ———— PR 对比图 ————
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (name, r) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(r['rec'], r['prec'], color=color, lw=2,
                label=f"{name} (AP={r['ap']:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curve Comparison (Macro-Average)', fontsize=14)
    ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, 'pr_comparison.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"PR 对比图已保存: {pr_path}")

    # ———— 准确率柱状图 ————
    names = list(results.keys())
    accs  = [results[n]['acc'] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars = ax.bar(names, accs, color=bar_colors, edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(min(accs) - 0.03, max(accs) * 1.08)
    ax.set_ylabel('Accuracy'); ax.set_title('Accuracy Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    bar_path = os.path.join(out_dir, 'accuracy_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"准确率柱状图已保存: {bar_path}")
