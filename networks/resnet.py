'''
the input size is 224x224.
this resnet structure is based on ResNet50
'''

import torch
import torch.nn as nn
import numpy as np

def Conv1(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=4):
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels*self.expansion),
        )
        # self.cv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.cv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
        #               bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.cv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1,
        #               bias=False),
        #     nn.BatchNorm2d(out_channels * self.expansion),
        # )

        if self.downsampling:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottle_neck(x)
        # out = self.cv1(x)
        # print('1')
        # out = self.cv2(out)
        # print('2')
        # out = self.cv3(out)
        # print('3')

        if self.downsampling:
            residual = self.downsampling(x)
        # print(out.shape)
        # print(residual.shape)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=196, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_channels=1, out_channels=64)  # (64x112x112)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64x56x56)

        self.layer1 = self.make_layer(in_channels=64, out_channels=64, block=blocks[0], stride=1)  # (256x56x56)
        self.layer2 = self.make_layer(in_channels=256, out_channels=128, block=blocks[1], stride=2)  # (512x28x28(stride=2))
        self.layer3 = self.make_layer(in_channels=512, out_channels=256, block=blocks[2], stride=2)  # (1024x14x14)
        # self.layer4 = self.make_layer(in_channels=1024, out_channels=512, block=blocks[3], stride=2)  # (2048x7x7)
        #
        # self.avg_pool = nn.AvgPool2d(7, stride=1)  # Input: (N, C, H_{in}, W_{in})
        #
        # self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, block, stride):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, downsampling=True))
        for i in range(1, block):
            layers.append(BottleNeck(out_channels*self.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)  # (64x112x112)
        x1_pool = self.max_pool(x1)  #(64x56x56)
        x2 = self.layer1(x1_pool)  # (256x56x56)
        x3 = self.layer2(x2)  # (512x28x28)
        x4 = self.layer3(x3)  # (1024x14x14)
        # x5 = self.layer4(x4)  # (2048x7x7)
        #
        # x = self.avg_pool(x5)  # (2048x1x1)
        # x = x.view(x.size(0), -1)  # x.size(0) == batch_size (b, 2048)
        # x = self.fc(x)
        return x1, x2, x3, x4


# input = torch.randn(1, 1, 224, 224)
# m = ResNet([3, 4, 6, 3])
# res, list = m.forward(input)
# print(res.shape)
# print(list[0].shape)
# print(list[1].shape)
# print(list[2].shape)