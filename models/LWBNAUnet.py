import torch
import torch.nn as nn

class LWBNAUnet(nn.Modules):
    def __init__(self, in_channel, out_channle):
        super(LWBNAUnet, self).__init__()
        