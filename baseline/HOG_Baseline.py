"""
baseline/HOG_Baseline.py
HOG + 颜色直方图特征提取 + LogisticRegression 四分类
纯传统 CV/ML 基线，零神经网络依赖
可独立运行，也可被 compare_models.py 导入
author: yukun-hh
date: 2026-5-14
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ★★★ 可配置参数 ★★★
# ============================================================
DATA_ROOT   = '../../trash_division_data/ultimate_4_class/'
IMAGE_SIZE  = 128
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS  = 32
# ============================================================

CLASS_NAMES = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']
NUM_CLASSES = 4


def extract_hog_color(image):
    img = image.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float64) / 255.0

    hog_feat = hog(arr, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   channel_axis=2, feature_vector=True)

    color_feat = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=COLOR_BINS, range=(0, 1))
        color_feat.append(hist)
    color_feat = np.concatenate(color_feat)

    return np.concatenate([hog_feat, color_feat])


class HOGLRBaseline:
    def __init__(self, data_root=DATA_ROOT, image_size=IMAGE_SIZE):
        self.data_root = data_root
        self.image_size = image_size
        self.clf = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1,
        )

    def _load_data(self, split):
        dir_path = os.path.join(self.data_root, split)
        features, labels = [], []
        for class_id in range(1, NUM_CLASSES + 1):
            class_dir = os.path.join(dir_path, str(class_id))
            if not os.path.isdir(class_dir):
                continue
            files = sorted(os.listdir(class_dir))
            for fname in tqdm(files, desc=f'{split}/class_{class_id}'):
                fpath = os.path.join(class_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        feat = extract_hog_color(img)
                    features.append(feat)
                    labels.append(class_id - 1)
                except Exception:
                    pass
        print(f"  {split}: {len(features)} 张")
        return np.array(features, dtype=np.float32), np.array(labels)

    def fit(self, train_dir=None):
        if train_dir is None:
            train_dir = 'train'
        print("  提取训练集 HOG 特征 ...")
        X, y = self._load_data(train_dir)
        self.clf.fit(X, y)

    def predict(self, val_dir=None):
        if val_dir is None:
            val_dir = 'val'
        print("  提取验证集 HOG 特征 ...")
        X, y = self._load_data(val_dir)
        preds = self.clf.predict(X)
        probs = self.clf.predict_proba(X)
        return y, preds, probs


# ============================================================
# compare_models.py 导入接口
# ============================================================

def get_hog_lr_preds(train_loader, val_loader, device):
    baseline = HOGLRBaseline()
    baseline.fit('train')
    return baseline.predict('val')


# ============================================================
# 独立运行入口
# ============================================================

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("HOG + LogisticRegression 基线")
    baseline = HOGLRBaseline()

    baseline.fit('train')
    y_true, y_preds, y_probs = baseline.predict('val')

    acc = accuracy_score(y_true, y_preds)
    macro_f1 = f1_score(y_true, y_preds, average='macro')
    print(f"\n验证集 Accuracy: {acc:.4f}")
    print(f"验证集 Macro-F1: {macro_f1:.4f}")
    print(f"\n分类报告:\n{classification_report(y_true, y_preds, target_names=CLASS_NAMES)}")

    cm = confusion_matrix(y_true, y_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', values_format='d', xticks_rotation=30)
    ax.set_title('HOG + LogisticRegression 混淆矩阵', fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'hog_lr_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存: {cm_path}")
