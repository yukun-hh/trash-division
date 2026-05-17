"""
baseline/VGG_KNN.py
VGG16 预训练模型特征提取 + KNN 四分类基线
可独立运行，也可被 compare_models.py 导入复用
author: yukun-hh
date: 2026-5-14
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class VGGKNNBaseline:
    def __init__(self, k=5, device='cpu',
                 data_root='../trash_division_data/ultimate_4_class/',
                 image_size=256, batch_size=32, num_workers=4):
        self.k = k
        self.device = device
        self.data_root = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extractor = load_vgg16_extractor(device)
        self.knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    def _get_loader(self, split):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        dataset = RobustImageFolder(
            root=os.path.join(self.data_root, split),
            transform=transform,
        )
        print(f"  {split}: {len(dataset)} 张")
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, drop_last=False)

    def fit(self, train_loader=None):
        if train_loader is None:
            train_loader = self._get_loader('train')
        print("  提取训练集特征 ...")
        train_feats, train_labels = extract_features(self.extractor, train_loader, self.device)
        self.knn.fit(train_feats, train_labels)

    def predict(self, val_loader=None):
        if val_loader is None:
            val_loader = self._get_loader('val')
        print("  提取验证集特征 ...")
        val_feats, val_labels = extract_features(self.extractor, val_loader, self.device)
        preds = self.knn.predict(val_feats)
        probs = self.knn.predict_proba(val_feats)
        return val_labels, preds, probs


if __name__ == '__main__':
    DATA_ROOT   = '../trash_division_data/ultimate_4_class/'
    BATCH_SIZE  = 32
    IMAGE_SIZE  = 256
    NUM_WORKERS = 4
    K           = 5

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    baseline = VGGKNNBaseline(k=K, device=device,
                               data_root=DATA_ROOT, image_size=IMAGE_SIZE,
                               batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    train_loader = baseline._get_loader('train')
    val_loader = baseline._get_loader('val')

    baseline.fit(train_loader)
    y_true, y_preds, y_probs = baseline.predict(val_loader)

    acc = accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average='macro')
    print(f"\n验证集 Accuracy: {acc:.4f}")
    print(f"验证集 Macro-F1: {macro_f1:.4f}")
    print(f"\n分类报告:\n{classification_report(y_true, y_preds, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', values_format='d', xticks_rotation=30)
    ax.set_title(f'Baseline Confusion Matrix (VGG16 + KNN, K={K})', fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg_knn_confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存: {out_path}")
