"""
模型定义文件 - ResNet-34
author : yukun-hh
date : 2026-4-10
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


class BasicBlock(nn.Module):
    """
    ResNet-34 基础残差块：3x3 -> 3x3
    若需要下采样或通道变化，则在跳跃连接中使用 1x1 卷积
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Net(nn.Module):

    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers_config = [
            (3, 64, 1),    # layer1
            (4, 128, 2),   # layer2
            (6, 256, 2),   # layer3
            (3, 512, 2),   # layer4
        ]

        self.in_channels = 64
        self.layer1 = self._make_layer(layers_config[0])
        self.layer2 = self._make_layer(layers_config[1])
        self.layer3 = self._make_layer(layers_config[2])
        self.layer4 = self._make_layer(layers_config[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, config):
        num_blocks, out_channels, stride = config
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Net(num_classes=4)
    summary(model, input_size=(3, 256, 256))
