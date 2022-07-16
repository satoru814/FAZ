import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, padding="same")
        self.norm = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x


class DoubleConv_Atten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv_Atten, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)
        self.pool = nn.MaxPool2d(2)
        self.atten = AttentionBlock(out_channel)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_skip = self.atten(x)
        x = self.pool(x_skip)
        return x, x_skip


class DoubleConv_Atten_Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv_Atten_Up, self).__init__()
        self.conv1 = ConvBlock(in_channel, out_channel)
        self.conv2 = ConvBlock(out_channel, out_channel)
        self.atten = AttentionBlock(out_channel)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.atten(x)
        out = self.up(x)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, in_channel):
        super(AttentionBlock, self,).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        residual = x
        x = self.pool(x)
        x = self.dense(x)
        return x * residual


class MiddleBlock(nn.Module):
    def __init__(self, in_channel):
        super(MiddleBlock, self).__init__()
        self.middleblock = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, padding="same"),
                                         AttentionBlock(in_channel),
                                         nn.Conv2d(in_channel, in_channel//2, 3, 1, padding="same"),
                                         AttentionBlock(in_channel//2),
                                         nn.Conv2d(in_channel//2, in_channel//4, 3, 1, padding="same"),
                                         AttentionBlock(in_channel//4),
                                         nn.Conv2d(in_channel//4, in_channel//8, 3, 1, padding="same"),
                                         AttentionBlock(in_channel//8),
                                         nn.Conv2d(in_channel//8, in_channel, 3, 1, padding="same"),
                                         )
    def forward(self,x):
        out = self.middleblock(x)
        return out


class LWBNAUnet(nn.Module):
    def __init__(self, in_channel, out_channel, nb_filter = 128):
        super(LWBNAUnet, self).__init__()
        self.conv1_1 = DoubleConv_Atten(in_channel, nb_filter)
        self.conv2_1 = DoubleConv_Atten(nb_filter, nb_filter)
        self.conv3_1 = DoubleConv_Atten(nb_filter, nb_filter)
        self.conv4_1 = DoubleConv_Atten(nb_filter, nb_filter)
        self.conv5 = DoubleConv_Atten_Up(nb_filter, nb_filter)
        self.conv2_2 = DoubleConv_Atten_Up(nb_filter, nb_filter)
        self.conv3_2 = DoubleConv_Atten_Up(nb_filter, nb_filter)
        self.conv4_2 = DoubleConv_Atten_Up(nb_filter, nb_filter)
        self.dropout = nn.Dropout2d(0.3)
        self.middle = MiddleBlock(nb_filter)
        self.finalblock = nn.Sequential(nn.Conv2d(nb_filter, nb_filter, 3, 1, padding="same"),
                                        nn.Conv2d(nb_filter, out_channel, 3, 1, padding="same"),
                                        AttentionBlock(out_channel),
                                        nn.Sigmoid())
    def forward(self, x):
        x1_1, x1_1_skip = self.conv1_1(x)
        x2_1, x2_1_skip = self.conv2_1(self.dropout(x1_1))
        x3_1, x3_1_skip = self.conv3_1(self.dropout(x2_1))
        x4_1, x4_1_skip = self.conv4_1(self.dropout(x3_1))
        x_middle = self.middle(self.dropout(x4_1))
        x5 = self.conv5(x_middle)
        x4_2 = self.conv4_2(self.dropout(x4_1_skip + x5))
        x3_2 = self.conv3_2(self.dropout(x3_1_skip + x4_2))
        x2_2 = self.conv2_2(self.dropout(x2_1_skip + x3_2))
        out = self.finalblock(self.dropout(x1_1_skip + x2_2))
        return out