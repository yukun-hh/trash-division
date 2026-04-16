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
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条（可选）
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [Train]')

    for images, labels in pbar:
        # 将数据移到 GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条信息
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度，节省内存
        for images, labels in tqdm(val_loader, desc='[Validate]'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """主训练函数"""

    # 1. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类用交叉熵

    # 或者使用 SGD + 动量
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # 学习率调度器（可选，帮助收敛）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 2. 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    # 3. 开始训练
    for epoch in range(epochs):
        print(f'\n{"=" * 50}')
        print(f'Epoch {epoch + 1}/{epochs}')

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device, epoch)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印结果
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ 保存最佳模型 (Acc: {val_acc:.2f}%)')

    # 4. 绘制训练曲线

    print(f'\n{"=" * 50}')
    print(f'训练完成！最佳验证准确率: {best_val_acc:.2f}%')

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
    model = model.to(device)

    # 打印模型信息
    print(f'Device: {device}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 2. 开始训练
    trained_model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=0.001,
        device=device
    )

    # 3. 加载最佳模型用于预测
    model.load_state_dict(torch.load('best_model.pth'))
    print('训练完成，最佳模型已加载')