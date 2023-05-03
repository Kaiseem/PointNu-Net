import torch
from torch import nn
from torchvision.ops import DeformConv2d, deform_conv2d
import math
class DeformConv(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1,groups=1, bias=None):
        super(DeformConv,self).__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        channels_ = groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)

class DeformConv123(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=None):
        super(DeformConv, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        channels_ = self.groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                     channels_,
                                     kernel_size=self.groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        nn.init.constant_(self.conv_offset.weight, 0)
        self.conv_offset.register_backward_hook(self._set_lr)
        self.conv_mask = nn.Conv2d(self.in_channels,
                                   self.groups * 1 * self.kernel_size[0] * self.kernel_size[1],
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   bias=True)
        nn.init.constant_(self.conv_mask.weight, 0)
        self.conv_mask.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def init_offset(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.uniform_(-stdv, stdv)
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        mask = torch.sigmoid(self.conv_mask(input))
        return deform_conv2d(input, offset, self.weight, self.bias, stride=self.stride,
                             padding=self.padding, dilation=self.dilation, mask=mask)