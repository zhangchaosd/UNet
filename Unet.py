import os
from Dataset import SEGData
import torch
import torch.nn as nn

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class DownsampleLayer(nn.modules):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.Conv_BN_RELU = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.Downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2 ) #??????????????
        )
    def forward(self, x):
        out1 = self.Conv_BN_RELU(x)
        out2 = self.Downsample(out1)
        print(out1.shape)
        #TODO out1 crop   copy????????  torchvision.transforms.CenterCrop(size)
        return out1, out2

class UpsampleLayer(nn.modules):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.Upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(in_channels = in_channel, out_channels = in_channel // 2, kernel_size = 1),
            nn.ReLU(inplace = True) #!!!!!!!!!!!!!!
        )
        self.Conv_BN_RELU = nn.Sequential(
            nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channel, out_channels = out_channel, kernel_size = 3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x, x2):
        out1 = self.Upsample(x)
        out2 = torch.cat((x2, out1), dim = 1) ###########################
        out3 = self.Conv_BN_RELU(out2)
        return out3

class UNet(nn.modules):
    def __init__(self, in_channel, out_channel):
        super().__init__() #TODO init params!!!!!!!!!!!!!!!!!!!!!!
        self.d1 = DownsampleLayer(in_channel, 64)
        self.d2 = DownsampleLayer(64, 128)
        self.d3 = DownsampleLayer(128, 256)
        self.d4 = DownsampleLayer(256, 512)
        self.Conv_BN_RELU_2 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.u1 = UpsampleLayer(1024 + 512, 512)
        self.u2 = UpsampleLayer(512 + 256, 256)
        self.u3 = UpsampleLayer(256 + 128, 128)
        self.u4 = UpsampleLayer(128 + 64, 64)
        self.Conv_Sigmoid = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = out_channel, kernel_size = 1, padding = 1),
            nn.Sigmoid()
        )
        #TODO    params INIT

    def forward(self, x):
        out1, x = self.d1(x)
        out2, x = self.d2(x)
        out3, x = self.d3(x)
        out4, x = self.d4(x)
        x = self.Conv_BN_RELU_2(x)
        x = self.u1(x, out4)
        x = self.u2(x, out3)
        x = self.u3(x, out2)
        x = self.u4(x, out1)
        x = self.Conv_Sigmoid(x)
        return x
