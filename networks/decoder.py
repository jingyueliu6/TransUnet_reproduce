import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):  # skip_channels is from encoder
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)  # same size
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#     config.decoder_channels = (256, 128, 64, 16)
class DecoderCup(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        head_channels = 512
        self.conv_more = Conv2dReLU(hidden_size, head_channels, kernel_size=3, padding=1)
        decoder_channels = (256, 128, 64, 14)
        in_channels = [head_channels] + list(decoder_channels[:-1])  # [512, 256, 128, 64]
        out_channels = decoder_channels  # (256, 128, 64, 16)

        skip_channels = [512, 256, 64]

        self.d1 = DecoderBlock(in_channels[0], out_channels[0], skip_channels[0])
        self.d2 = DecoderBlock(in_channels[1], out_channels[1], skip_channels[1])
        self.d3 = DecoderBlock(in_channels[2], out_channels[2], skip_channels[2])
        self.d4 = DecoderBlock(in_channels[2], out_channels[2], 0)

    def forward(self, hidden_states, x1, x2, x3):
        #  hidden_states (b, hidden_size(D), 14, 14)
        x = self.conv_more(hidden_states)
        x = self.d1(x, x3)
        x = self.d2(x, x2)
        x = self.d3(x, x1)
        x = self.d4(x, None)
        return x
