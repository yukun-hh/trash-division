"""
Baseline.py
VGG16 预训练模型特征提取 + KNN 四分类基线
author: yukun-hh
date: 2026-5-14
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

from Dataloader import RobustImageFolder

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ★★★ 可配置参数 ★★★
# ============================================================
DATA_ROOT   = '../trash_division_data/ultimate_4_class/'
BATCH_SIZE  = 32
IMAGE_SIZE  = 256
NUM_WORKERS = 4
K           = 5
# ============================================================

CLASS_NAMES = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']


def load_vgg16_extractor(device):
    try:
        model = models.vgg16(weights='IMAGENET1K_V1')
    except TypeError:
        model = models.vgg16(pretrained=True)
    model.classifier = nn.Identity()
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Extracting features'):
            images = images.to(device)
            feats = model(images)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


if __name__ == '__main__':
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RobustImageFolder(
        root=os.path.join(DATA_ROOT, 'train'),
        transform=val_transform,
    )
    val_dataset = RobustImageFolder(
        root=os.path.join(DATA_ROOT, 'val'),
        transform=val_transform,
    )
    print(f"训练集: {len(train_dataset)}  验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    extractor = load_vgg16_extractor(device)
    print("VGG16 特征提取器加载完成 (classifier 已移除)")

    print("提取训练集特征 ...")
    train_feats, train_labels = extract_features(extractor, train_loader, device)
    print(f"训练特征: {train_feats.shape}")

    print("提取验证集特征 ...")
    val_feats, val_labels = extract_features(extractor, val_loader, device)
    print(f"验证特征: {val_feats.shape}")

    knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
    knn.fit(train_feats, train_labels)
    print(f"KNN (K={K}) 训练完成")

    val_preds = knn.predict(val_feats)

    acc = accuracy_score(val_labels, val_preds)
    macro_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"\n验证集 Accuracy: {acc:.4f}")
    print(f"验证集 Macro-F1: {macro_f1:.4f}")
    print(f"\n分类报告:\n{classification_report(val_labels, val_preds, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', values_format='d', xticks_rotation=30)
    ax.set_title(f'Baseline Confusion Matrix (VGG16 + KNN, K={K})', fontsize=14)
    plt.tight_layout()
    plt.savefig('baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("混淆矩阵已保存: baseline_confusion_matrix.png")
