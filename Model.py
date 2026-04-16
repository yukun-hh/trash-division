"""
这个文件是模型的定义文件，请不要擅自修改，如有疑问微信群里反馈
单独运行本文件将会输出模型结构
目前的话是一个36层的模型，模型总量应该是在80M左右 如果到时候还是欠拟合的话再考虑去做更深的结构
author : yukun-hh
date : 2026-4-10

"""
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary


# 残差块
class Resblock(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        """
        :param input_channels: 进入残差块时的原通道
        :param output_channels: 输出时的通道数
        :param use_1x1conv: 如果输入和输出通道不相等时，需要用一个1x1的卷积层对原来的输入进行一个通道提升
        :param strides: 默认1，如果大于1起到缩小张量的作用
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class Net(nn.Module):
    """
    模型的主要结构就在这里了，到时也好该和调用
    现在必须实现的方法：
    目前还是以图片缩放到256＊256构建残差块
    """

    def __init__(self):
        super().__init__()

        # 定义残差块的辅助方法
        def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
            """
            :param input_channels: 输入维度
            :param num_channels: 输出维度
            :param num_residuals: 单个残差层的残差块数
            :param first_block: 第一块不用下采样 特殊控制
            :return: list[nn.Module]
            """
            blk = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(Resblock(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(Resblock(num_channels, num_channels))
            return blk

        # 构建网络各层
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        """
        7×7 卷积层，输出通道 64，步长 2，填充 3
        (3×256×256)->(64×128×128)
        批归一化 relu层 
        最大池化 
        (64×128×128)->(64×64×64)
        """
        self.b2 = nn.Sequential(*resnet_block(64, 64, num_residuals=3, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, num_residuals=4))
        self.b4 = nn.Sequential(*resnet_block(128, 256, num_residuals=6))
        self.b5 = nn.Sequential(*resnet_block(256, 512, num_residuals=3))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = Net()
    # 使用 torchsummary 查看模型结构
    summary(model, input_size=(3, 256, 256))