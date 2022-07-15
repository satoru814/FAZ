import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from . import modules


class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv0_0 = modules.VGGBlock(in_channel, nb_filter[0], nb_filter[0])
        self.conv1_0 = modules.VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = modules.VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = modules.VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = modules.VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv3_1 = modules.VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2_2 = modules.VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv1_3 = modules.VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = modules.VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))
        out = self.final(x0_4)
        out = self.sigmoid(out)
        return out
