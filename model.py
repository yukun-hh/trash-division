"""
这个文件是模型的定义文件，请不要擅自修改，如有疑问微信群里反馈
单独运行本文件将会输出模型结构
目前的话是一个36层的模型，模型总量应该是在10M左右 如果到时候还是欠拟合的话再考虑去做更深的结构
author : yukun-hh
date : 2026-4-10

"""
#神经网络模型库
import torch
from torch import nn
from torch.nn import functional as F

#残差块
class Resblock(nn.Module):
    def __init__(self, input_channels,output_channels,use_1x1conv=False,strides=1):
        """

        :param input_channels: 进入残差块时的原通道
        :param output_channels: 输出时的通道数
        :param use_1x1conv: 如果输入和输出通道不相等时，需要用一个1x1的卷积层对原来的输入进行一个通道提升
        :param strides: 默认1，如果大于1起到缩小张量的作用
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1,stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Net():
    """
    模型的主要结构就在这里了，到时也好该和调用
    现在必须实现的方法：
    目前还是以图片缩放到256＊256构建残差块
    """
    net = nn.Sequential()
    def resnet_block(self,input_channels, num_channels, num_residuals,
                     first_block=False):
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
                blk.append(Resblock(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
            else:
                blk.append(Resblock(num_channels, num_channels))
        return blk
    def __init__(self):
        b1 = nn.Sequential( nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                            nn.BatchNorm2d(64), nn.ReLU(),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            )
        """
        7×7 卷积层，输出通道 64，步长 2，填充 3
        (3×256×256)->(64×128×128)
        批归一化 relu层 
        最大池化 
        (64×128×128)->(64×64×64)
        """
        b2 = nn.Sequential(*self.resnet_block(64, 64, num_residuals=3, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, num_residuals=4))
        b4 = nn.Sequential(*self.resnet_block(128, 256, num_residuals=6))
        b5 = nn.Sequential(*self.resnet_block(256, 512, num_residuals=3))
        self.net = nn.Sequential(b1, b2, b3, b4, b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 4))
    def get_network(self):
        return self.net

"""
Sequential(
  (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
)
Sequential output shape:	torch.Size([1, 64, 64, 64])
Sequential(
  (0): Resblock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): Resblock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (2): Resblock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
Sequential output shape:	torch.Size([1, 64, 64, 64])
Sequential(
  (0): Resblock(
    (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): Resblock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (2): Resblock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (3): Resblock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
Sequential output shape:	torch.Size([1, 128, 32, 32])
Sequential(
  (0): Resblock(
    (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): Resblock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (2): Resblock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (3): Resblock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (4): Resblock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (5): Resblock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
Sequential output shape:	torch.Size([1, 256, 16, 16])
Sequential(
  (0): Resblock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (1): Resblock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (2): Resblock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
Sequential output shape:	torch.Size([1, 512, 8, 8])
AdaptiveAvgPool2d(output_size=(1, 1))
AdaptiveAvgPool2d output shape:	torch.Size([1, 512, 1, 1])
Flatten(start_dim=1, end_dim=-1)
Flatten output shape:	torch.Size([1, 512])
Linear(in_features=512, out_features=4, bias=True)
Linear output shape:	torch.Size([1, 4])
"""

if __name__ == '__main__':
    Net_new = Net()
    X = torch.rand(size=(1, 3, 256, 256))
    for layer in Net_new.get_network():
        print(layer)
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)








