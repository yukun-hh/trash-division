"""
目前是一份数据加载用的代码，没有调整，因为现在还没有配置好数据集
这个文件目前还不能运行！！！


author:yukun-hh
date ：2026-4-10
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_dataloaders(data_root='..',
                       batch_size=32,
                       image_size=256,
                       val_split=0.2,
                       num_workers=4,
                       augment=True):
    """
    创建训练和验证的 DataLoader

    Args:
        data_root: 项目根目录（包含 train 和 val 文件夹）
        batch_size: 批次大小
        image_size: 统一缩放的尺寸（256x256）
        val_split: 从训练集中划分验证集的比例（如果你没有独立的 val 文件夹）
        num_workers: 数据加载线程数
        augment: 是否使用数据增强

    Returns:
        train_loader, val_loader, class_names
    """

    # 1. 定义图像预处理（转换）流程
    # ==================================

    # 训练时的数据增强（提高泛化能力）
    train_transform = transforms.Compose([
        # 随机调整大小（保留长宽比后裁剪）
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),

        # 随机水平翻转（对于垃圾分拣，翻转后类别不变）
        transforms.RandomHorizontalFlip(p=0.5),

        # 随机旋转（±15度）
        transforms.RandomRotation(degrees=15),

        # 随机亮度/对比度调整（模拟不同光照）
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # 转换为张量
        transforms.ToTensor(),

        # 标准化（使用 ImageNet 的均值标准差，可改为自己数据集的）
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证时的预处理（只做必要的操作）
    val_transform = transforms.Compose([
        # 直接缩放到固定大小
        transforms.Resize((image_size, image_size)),

        # 转换为张量
        transforms.ToTensor(),

        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集
    # ==================================
    print("使用独立的 val 文件夹")
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, 'train'),
        transform=train_transform if augment else val_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, 'val'),
        transform=val_transform
    )
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")


    # 3. 创建 DataLoader
    # ==================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱顺序
        num_workers=num_workers,
        pin_memory=True,  # 加速 GPU 传输
        drop_last=True  # 丢弃最后一个不完整的 batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 4. 获取类别名称
    class_names = train_dataset.classes if hasattr(train_dataset, 'classes') else ['0', '1', '2', '3']
    print(f"类别: {class_names}")
    print(f"类别映射: {train_dataset.class_to_idx if hasattr(train_dataset, 'class_to_idx') else '0-3'}")

    return train_loader, val_loader, class_names


# ========== 辅助函数：检查数据加载是否正确 ==========

def visualize_batch(dataloader, class_names, num_images=8):
    """可视化一个 batch 的图像，检查数据是否正确"""
    images, labels = next(iter(dataloader))

    # 反标准化（用于显示）
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(1, min(num_images, len(images)), figsize=(15, 3))
    if len(images) == 1:
        axes = [axes]

    for i in range(min(num_images, len(images))):
        img = images[i].cpu()
        img = img * std + mean  # 反标准化
        img = torch.clamp(img, 0, 1)  # 裁剪到 [0,1]
        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # 打印批次信息
    print(f"Batch 图像形状: {images.shape}")
    print(f"Batch 标签: {labels}")
    print(f"标签分布: {torch.bincount(labels)}")





# ========== 使用示例 ==========

if __name__ == '__main__':
    train_loader, val_loader, class_names = create_dataloaders(
        data_root='..',  # 与trash-division同级文件夹
        batch_size=32,  # 根据你的显存调整
        image_size=256,  # 与你模型输入一致
        num_workers=4,  # Windows 可能需设为 0
        augment=True  # 训练时使用数据增强
    )
    visualize_batch(train_loader, class_names, num_images=8)
