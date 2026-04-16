"""
模型定义文件 - 使用瓶颈结构 (Bottleneck) 的深度残差网络
目标：约50层，参数量约80M
author : yukun-hh
date : 2026-4-10
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class Bottleneck(nn.Module):
    """
    瓶颈残差块：1x1(降维) -> 3x3 -> 1x1(升维)
    若需要下采样或通道变化，则在跳跃连接中使用1x1卷积
    """
    expansion = 4  # 输出通道是中间通道的4倍

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        """
        :param in_channels: 输入通道数
        :param mid_channels: 中间层通道数（1x1降维后的通道数）
        :param stride: 步长，用于下采样
        :param downsample: 下采样模块（当stride≠1或通道变化时使用）
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):
    """
    基于 Bottleneck 的 ResNet 风格模型
    各阶段配置仿照 ResNet-50，适当调整宽度以达到约80M参数
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # 第一阶段：7x7卷积 + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差阶段定义
        # 每个阶段的参数：[块数, 中间通道数, 步长]
        # 为了达到80M参数，我们略微加宽网络（相比标准ResNet-50）
        layers_config = [
            (3, 64, 1),    # stage2: 3个瓶颈块，输出通道 64*4=256
            (4, 128, 2),   # stage3: 4个瓶颈块，输出通道 128*4=512
            (14, 256, 2),  # stage4: 14个瓶颈块，输出通道 256*4=1024（加深至此阶段）
            (3, 512, 2)    # stage5: 3个瓶颈块，输出通道 512*4=2048
        ]

        self.in_channels = 64
        self.stage2 = self._make_layer(layers_config[0])
        self.stage3 = self._make_layer(layers_config[1])
        self.stage4 = self._make_layer(layers_config[2])
        self.stage5 = self._make_layer(layers_config[3])

        # 全局池化与分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, config):
        """
        构建一个残差阶段
        :param config: (块数, 中间通道数, 第一阶段步长)
        :return: nn.Sequential
        """
        num_blocks, mid_channels, stride = config
        downsample = None
        layers = []

        # 第一个块可能需要下采样和通道匹配
        if stride != 1 or self.in_channels != mid_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * Bottleneck.expansion),
            )

        layers.append(
            Bottleneck(self.in_channels, mid_channels, stride, downsample)
        )
        self.in_channels = mid_channels * Bottleneck.expansion

        # 后续块
        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(self.in_channels, mid_channels)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Net(num_classes=4)
    summary(model, input_size=(3, 256, 256))