# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F



BatchNorm2d = nn.BatchNorm2d #functools.partial(InPlaceABNSync, activation='none')
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, _blocks, _num_channels, _num_modules, _num_branches, _num_blocks, _fuse_method):
        #extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()
        self.num_branches = _num_branches
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        num_channels, block, num_blocks = _num_channels[0][0], _blocks[0], _num_blocks[0]

        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks[0])
        stage1_out_channel = block.expansion * num_channels

        num_channels, block = _num_channels[1], _blocks[1]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(_num_modules[1], _num_branches[1], _num_blocks[1], _num_channels[1], block, _fuse_method[1], num_channels)


        num_channels, block= _num_channels[2], _blocks[2]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels =  self._make_stage(_num_modules[2], _num_branches[2], _num_blocks[2], _num_channels[2], block, _fuse_method[2], num_channels)

        num_channels, block= _num_channels[3], _blocks[3]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels =self._make_stage(_num_modules[3], _num_branches[3], _num_blocks[3], _num_channels[3], block, _fuse_method[3], num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, block,
                    fuse_method, num_inchannels, multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.num_branches[1]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.num_branches[2]):
            if self.transition2[i] is not None:
                if i < self.num_branches[1]:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.num_branches[3]):
            if self.transition3[i] is not None:
                if i < self.num_branches[2]:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def hrnet_w18_v2(pretrained=False, **kwargs):
    blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock]
    num_modules = [1, 1, 3, 2]
    num_branches = [1, 2, 3, 4]
    num_blocks = [[2], [2, 2], [2, 2, 2], [2, 2, 2, 2]]
    num_channels = [[64], [18, 36], [18, 36, 72], [18, 36, 72, 144]]
    fuse_method = ['sum', 'sum', 'sum', 'sum']
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    if pretrained:
        root = r'C:\Users\Ykk\.torch\models\hrnet_w18_small_model_v2.pth'
        old_dict = torch.load(root)
        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                pass
                #print(k)
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def hrnet_w32(pretrained=False, **kwargs):
    blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock]
    num_modules = [1, 1, 4, 3]
    num_branches = [1, 2, 3, 4]
    num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
    num_channels = [[64], [32, 64], [32, 64, 128], [32, 64, 128, 256]]
    fuse_method = ['sum', 'sum', 'sum', 'sum']
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    if pretrained:
        root = r'C:\Users\Ykk\.torch\models\hrnetv2_w32_imagenet_pretrained.pth'
        old_dict = torch.load(root)
        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                pass
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def hrnet_w44(pretrained=False, **kwargs):
    blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock]
    num_modules = [1, 1, 4, 3]
    num_branches = [1, 2, 3, 4]
    num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
    num_channels = [[64], [44, 88], [44, 88, 176], [44, 88, 176, 352]]
    fuse_method = ['sum', 'sum', 'sum', 'sum']
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    if pretrained:
        root = r'C:\Users\Ykk\.torch\models\hrnetv2_w44_imagenet_pretrained.pth'
        old_dict = torch.load(root)
        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                pass
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def hrnet_w48(pretrained=False, **kwargs):
    blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock]
    num_modules = [1, 1, 4, 3]
    num_branches = [1, 2, 3, 4]
    num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
    num_channels = [[64], [48, 96], [48, 96, 192], [48, 96, 192,384]]
    fuse_method = ['sum', 'sum', 'sum', 'sum']
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    if pretrained:
        root = r'.torch\models\hrnetv2_w48_imagenet_pretrained.pth'
        old_dict = torch.load(root)
        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                pass
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def hrnet_w64(pretrained=False, **kwargs):
    blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock]
    num_modules = [1, 1, 4, 3]
    num_branches = [1, 2, 3, 4]
    num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
    num_channels = [[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]
    fuse_method = ['sum', 'sum', 'sum', 'sum']
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)
    if pretrained:
        root = r'.torch\models\hrnetv2_w64_imagenet_pretrained.pth'
        old_dict = torch.load(root)
        model_dict = model.state_dict()
        for i,(k,v) in enumerate(old_dict.items()):
            if k not in model_dict:
                #print(k)
                pass
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    from torchsummaryX import summary
    model=hrnet_w18_v2(pretrained=True)
    #output=model(img)
    summary(model, torch.zeros((1, 3, 256, 256)))