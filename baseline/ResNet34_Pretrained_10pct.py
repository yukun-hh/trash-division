"""
baseline/ResNet34_Pretrained_10pct.py
ResNet-34 ImageNet 预训练权重 + 10% 训练集微调
可独立运行训练，也可被 compare_models.py 导入
author: yukun-hh
date: 2026-5-14
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import matplotlib

from Dataloader import RobustImageFolder

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# ★★★ 可配置参数 ★★★
# ============================================================
DATA_ROOT   = '../../trash_division_data/ultimate_4_class/'
BATCH_SIZE  = 32
IMAGE_SIZE  = 256
NUM_WORKERS = 4
EPOCHS      = 30
LR          = 0.001
TRAIN_PCT   = 0.1
SEED        = 42
DROPOUT     = 0.3
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet34_10pct.pth')
LOG_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet34_10pct_log.csv')
# ============================================================

NUM_CLASSES = 4
CLASS_NAMES = ['厨余垃圾', '可回收物', '其他垃圾', '有害垃圾']


class PretrainedResNet34(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.backbone = models.resnet34(weights='IMAGENET1K_V1')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def freeze_early_layers(self):
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False

    def print_trainable_info(self):
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"  冻结参数: {frozen:,}  可训练参数: {trainable:,}  ({100.*trainable/total:.1f}%)")


def compute_macro_f1(predicted, targets, num_classes=NUM_CLASSES):
    tp = torch.zeros(num_classes, device=predicted.device)
    fp = torch.zeros(num_classes, device=predicted.device)
    fn = torch.zeros(num_classes, device=predicted.device)
    for c in range(num_classes):
        tp[c] = ((predicted == c) & (targets == c)).sum()
        fp[c] = ((predicted == c) & (targets != c)).sum()
        fn[c] = ((predicted != c) & (targets == c)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.append(predicted)
        all_labels.append(labels)
        batch_f1 = compute_macro_f1(predicted, labels)
        pbar.set_postfix({'loss': loss.item(), 'F1': f'{batch_f1:.4f}',
                          'Acc': f'{100.*correct/total:.2f}%'})
    epoch_loss = running_loss / total
    epoch_f1 = compute_macro_f1(torch.cat(all_preds), torch.cat(all_labels))
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_f1, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc='[Validate]'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.append(predicted)
        all_labels.append(labels)
    epoch_loss = running_loss / total
    epoch_f1 = compute_macro_f1(torch.cat(all_preds), torch.cat(all_labels))
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_f1, epoch_acc


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_f1': [], 'train_acc': [],
               'val_loss': [], 'val_f1': [], 'val_acc': []}
    best_val_f1 = 0.0

    log_file = open(LOG_PATH, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'train_loss', 'train_f1', 'train_acc',
                         'val_loss', 'val_f1', 'val_acc', 'lr', 'best'])

    for epoch in range(epochs):
        print(f'\n{"="*50}')
        print(f'Epoch {epoch+1}/{epochs}')

        train_loss, train_f1, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_f1, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Macro-F1: {train_f1:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val   Macro-F1: {val_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        best_mark = ''
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_mark = 'best'
            print(f'✓ 保存最佳模型 (Macro-F1: {val_f1:.4f})')

        lr_val = optimizer.param_groups[0]['lr']
        log_writer.writerow([epoch+1, train_loss, train_f1, train_acc,
                             val_loss, val_f1, val_acc, lr_val, best_mark])
        log_file.flush()

    log_file.close()
    print(f'\n训练完成！最佳验证 Macro-F1: {best_val_f1:.4f}')
    return history


# ============================================================
# compare_models.py 导入接口
# ============================================================

def get_resnet34_10pct_preds(train_loader, val_loader, device):
    model = PretrainedResNet34(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    model = model.to(device).eval()

    y_true, y_preds, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='ResNet-34 (10%)'):
            images, labels = images.to(device), labels
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.append(labels.numpy())
            y_preds.append(preds.cpu().numpy())
            y_probs.append(probs.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_preds), np.concatenate(y_probs)


# ============================================================
# 独立训练入口
# ============================================================

if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_train_dataset = RobustImageFolder(
        root=os.path.join(DATA_ROOT, 'train'),
        transform=train_transform,
    )
    val_dataset = RobustImageFolder(
        root=os.path.join(DATA_ROOT, 'val'),
        transform=val_transform,
    )

    n_train = len(full_train_dataset)
    n_subset = max(1, int(n_train * TRAIN_PCT))
    indices = random.sample(range(n_train), n_subset)
    train_dataset = Subset(full_train_dataset, indices)
    print(f"训练集: {len(train_dataset)} / {n_train} ({TRAIN_PCT*100:.0f}%)")
    print(f"验证集: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, drop_last=False)

    model = PretrainedResNet34(num_classes=NUM_CLASSES, dropout=DROPOUT)
    model.freeze_early_layers()
    model.print_trainable_info()
    model = model.to(device)

    history = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    print(f"模型已保存: {MODEL_SAVE_PATH}")
    print(f"训练日志已保存: {LOG_PATH}")
