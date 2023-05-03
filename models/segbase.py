import torchvision
from .backbone.resnetv1b import resnet50_v1s,resnet101_v1s
from .backbone.res2netv1b import res2net50_v1b
from .backbone.resnext import resnext50_32x4d,resnext101_32x8d
from .backbone.seg_hrnet import hrnet_w18_v2,hrnet_w32,hrnet_w44,hrnet_w48,hrnet_w64
from .backbone.resnextdcn import resnext101_32x8d_dcn
from .backbone.swim_transformer import swim_large
from torch.nn.modules.batchnorm import _BatchNorm
__all__ = ['SegBaseModel']
import torch
import torch.nn as nn
import torch.nn.functional as F
class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation
    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
        resnest
        resnext
        res2net
        DLA
    """
    def __init__(self, backbone='enc', pretrained_base=False, frozen_stages=-1,norm_eval=False, **kwargs):
        super(SegBaseModel, self).__init__()
        self.norm_eval=norm_eval
        self.frozen_stages=frozen_stages
        if backbone == 'resnext101dcn':
            self.pretrained = resnext101_32x8d_dcn(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'hrnet18':
            self.pretrained = hrnet_w18_v2(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone =='hrnet32':
            self.pretrained=hrnet_w32(pretrained=pretrained_base,  **kwargs)
        elif backbone =='hrnet64':
            self.pretrained=hrnet_w64(pretrained=pretrained_base,  **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'resnext101':
            self.pretrained = resnext101_32x8d(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'res2net50':
            self.pretrained = res2net50_v1b(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnext50':
            self.pretrained = resnext50_32x4d(pretrained=pretrained_base, dilated=False, **kwargs)
        elif backbone == 'swin':
            self.pretrained = swim_large(pretrained=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self._train()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def freeze(self):
        self.set_requires_grad(self.pretrained,False)
        self.set_requires_grad([self.pretrained.conv1,self.pretrained.bn1],True)

    def unfreeze(self):
        self.set_requires_grad([self.pretrained],True)

    def _train(self, mode=True):
        super(SegBaseModel, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            print('Freeze backbone BN using running mean and std')
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.pretrained.bn1.eval()
            for m in [self.pretrained.conv1, self.pretrained.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
            if hasattr(self.pretrained, 'conv2'):
                self.pretrained.bn2.eval()
                for m in [self.pretrained.conv2, self.pretrained.bn2]:
                    for param in m.parameters():
                        param.requires_grad = False

        print(f'Freezing backbone stage {self.frozen_stages}')
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self.pretrained, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # trick: only train conv1 since not use ImageNet mean and std image norm
        if hasattr(self.pretrained,'conv1'):
            print('active train conv1 and bn1')
            self.set_requires_grad([self.pretrained.conv1,self.pretrained.bn1],True)

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        c1 = self.pretrained.maxpool(x)
        c2 = self.pretrained.layer1(c1)
        c3 = self.pretrained.layer2(c2)
        c4 = self.pretrained.layer3(c3)
        c5 = self.pretrained.layer4(c4)
        return c1, c2, c3, c4, c5




