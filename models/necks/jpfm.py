import torch
import torch.nn as nn

class JPFM(nn.Module):
    def __init__(self,in_channel,width=256):
        super(JPFM, self).__init__()

        self.out_channel = width*4
        self.dilation1 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True))
        self.dilation2 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True))
        self.dilation3 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True))
        self.dilation4 = nn.Sequential(
            nn.Conv2d(in_channel, width, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True))

    def forward(self,feat):
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        return feat

