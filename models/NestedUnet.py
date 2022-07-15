from re import X
import torch
from torch import nn
from . import modules


class NestedUNet(nn.Module):
    def __init__(self, in_channel, out_channel, deep_supervision=False, **kwargs):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision =deep_supervision
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = modules.VGGBlock(in_channel, nb_filter[0], nb_filter[0])
        self.conv1_0 = modules.VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = modules.VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = modules.VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = modules.VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = modules.VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = modules.VGGBlock(nb_filter[2]+nb_filter[1], nb_filter[1], nb_filter[1])
        self.conv2_1 = modules.VGGBlock(nb_filter[3]+nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3_1 = modules.VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])

        self.conv0_2 = modules.VGGBlock(nb_filter[0]+nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = modules.VGGBlock(nb_filter[1]+nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = modules.VGGBlock(nb_filter[2]+nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = modules.VGGBlock(nb_filter[0]+nb_filter[0]+nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = modules.VGGBlock(nb_filter[1]+nb_filter[1]+nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = modules.VGGBlock(nb_filter[0]+nb_filter[0]+nb_filter[0]+nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final(x0_1)
            output2 = self.final(x0_2)
            output3 = self.final(x0_3)
            output4 = self.final(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output



