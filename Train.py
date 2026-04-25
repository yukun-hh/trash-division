"""
目前是由AI先生成了一份训练用代码，没有调整，因为现在还没有设计好数据迭代器
这个文件目前还不能运行！！！

最佳模型将会保存在根目录下
author:yukun-hh
date ：2026-4-10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # 进度条，可选
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
    """训练一个epoch"""
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
    """验证函数"""
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


def compute_class_weights(dataset, num_classes=4, device='cpu'):
    class_counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        lbl = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[lbl] += 1
    total = class_counts.sum()
    weights = total / (num_classes * class_counts)
    return weights.to(device)


def train(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """主训练函数"""

    # 1. 定义损失函数和优化器
    class_weights = compute_class_weights(train_loader.dataset, num_classes=4, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # 多分类用交叉熵

    # 或者使用 SGD + 动量
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # 学习率调度器（可选，帮助收敛）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 2. 记录训练历史
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }

    best_val_f1 = 0.0

    # 3. 开始训练
    for epoch in range(epochs):
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch + 1}/{epochs}')

        # 训练
        train_loss, train_f1, train_acc = train_one_epoch(model, train_loader, criterion,
                                                          optimizer, device, epoch)

        # 验证
        val_loss, val_f1, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)

        # 打印结果
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Macro-F1: {train_f1:.4f}')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val   Macro-F1: {val_f1:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ 保存最佳模型 (Macro-F1: {val_f1:.4f})')

    # 4. 绘制训练曲线

    print(f'\n{"=" * 50}')
    print(f'训练完成！最佳验证 Macro-F1: {best_val_f1:.4f}')

    return model, history


# ========== 使用示例 ==========
if __name__ == '__main__':
    # 假设你的 dataloader 已经写好了
    train_loader, val_loader, class_names = create_dataloaders(
        data_root='../trash_division_data/ultimate_4_class/',  # 与trash-division同级文件夹
        batch_size=16,  # 根据你的显存调整
        image_size=256,  # 与你模型输入一致
        num_workers=8,  # Windows 可能需设为 0
        augment=True  # 训练时使用数据增强
    )

    # 1. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu')
    model = Net(num_classes=4)  # 根据你的 Net 类调整
    #断点继续训练
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth',map_location=torch.device('cpu')))
    model = model.to(device)

    # 打印模型信息
    print(f'Device: {device}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 2. 开始训练
    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=0.001,
        device=device
    )
    # 3. 加载最佳模型用于预测
    model.load_state_dict(torch.load('best_model.pth'))