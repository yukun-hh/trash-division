"""
微调脚本：冻结 conv1 + stage2，微调 stage3~fc
加大少样本类别的 loss 权重
author: yukun-hh
date ：2026-4-25
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from Model import Net
from Dataloader import create_dataloaders
import os


def compute_macro_f1(predicted, targets, num_classes=4):
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


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]')

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
        pbar.set_postfix({'loss': loss.item(), 'F1': f'{batch_f1:.4f}', 'Acc': f'{100. * correct / total:.2f}%'})

    epoch_loss = running_loss / total
    epoch_f1 = compute_macro_f1(torch.cat(all_preds), torch.cat(all_labels))
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_f1, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='[Validate]'):
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


def compute_class_weights(dataset, num_classes=4, device='cpu', power=1.0):
    class_counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        lbl = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[lbl] += 1
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    weights = weights ** power
    return weights.to(device)


def freeze_base_layers(model):
    frozen_layers = []
    for name, param in model.conv1.named_parameters():
        param.requires_grad = False
        frozen_layers.append(f'conv1.{name}')
    for name, param in model.bn1.named_parameters():
        param.requires_grad = False
        frozen_layers.append(f'bn1.{name}')
    for name, param in model.stage2.named_parameters():
        param.requires_grad = False
        frozen_layers.append(f'stage2.{name}')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'冻结层数: {len(frozen_layers)} 个参数组')
    print(f'可训练参数量: {trainable:,} / {total:,} ({100. * trainable / total:.1f}%)')
    return model


def finetune(model, train_loader, val_loader, epochs=30, lr=0.0001, device='cuda'):
    class_weights = compute_class_weights(train_loader.dataset, num_classes=4, device=device, power=1.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }

    best_val_f1 = 0.0

    for epoch in range(epochs):
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch + 1}/{epochs}')

        train_loss, train_f1, train_acc = train_one_epoch(model, train_loader, criterion,
                                                          optimizer, device, epoch)

        val_loss, val_f1, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Macro-F1: {train_f1:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val   Macro-F1: {val_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'finetuned_model.pth')
            print(f'✓ 保存最佳微调模型 (Macro-F1: {val_f1:.4f})')

    print(f'\n{"=" * 50}')
    print(f'微调完成！最佳验证 Macro-F1: {best_val_f1:.4f}')

    return model, history


if __name__ == '__main__':
    train_loader, val_loader, class_names = create_dataloaders(
        data_root='../trash_division_data/ultimate_4_class/',
        batch_size=16,
        image_size=256,
        num_workers=8,
        augment=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu')

    model = Net(num_classes=4)

    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
        print('✓ 加载预训练权重 best_model.pth')
    else:
        print('⚠ 未找到 best_model.pth，使用随机初始化权重')

    model = model.to(device)
    model = freeze_base_layers(model)

    print(f'Device: {device}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    trained_model, history = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=0.0001,
        device=device
    )

    model.load_state_dict(torch.load('finetuned_model.pth'))
