import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from ..base_ops.DCNv2 import DeformConv
__all__ = ['ResNext', 'resnext101_32x8d_dcn']

model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_dcn=False, **kwargs):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        if use_dcn:
            self.conv2 = DeformConv(width, width, kernel_size=3, stride=stride,
                             padding=dilation,dilation=dilation, groups=groups)

          #  self.conv2.conv_offset.weight.data.zero_()
           # self.conv2.conv_offset.bias.data.zero_()
        else:
            self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNext(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, norm_layer=nn.BatchNorm2d,deep_stem=False, **kwargs):
        super(ResNext, self).__init__()

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer,use_dcn=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer,use_dcn=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer,use_dcn=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer,use_dcn=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d,use_dcn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = list()
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, norm_layer=norm_layer,use_dcn=use_dcn))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, dilation=2, norm_layer=norm_layer,use_dcn=use_dcn))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=1, norm_layer=norm_layer,use_dcn=use_dcn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnext101_32x8d_dcn(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNext(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnext101_32x8d'])

        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                print(k)
                pass
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    model = resnext101_32x8d_dcn()