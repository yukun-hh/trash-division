"""
这个文件是模型的定义文件，请不要擅自修改，如有疑问微信群里反馈
author : yukun-hh
date : 2026-4-10

"""
#神经网络模型库
import torch
from modelscope.msdatasets.dataset_cls.custom_datasets.audio.kws_nearfield_processor import padding
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
        self.conv2 = nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1,stride=strides)
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
    def


