'''ResNeXt in PyTorch.
See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import logging
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint
from mmcv.cnn import (constant_init, kaiming_init)

__all__ = ['SK2NeXt29']

class ResNeXtBottleneck(nn.Module):
  expansion = 4
  """
  RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
  """
  def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
    super(ResNeXtBottleneck, self).__init__()

    D = int(math.floor(planes * (base_width/64.0)))
    C = cardinality

    self.conv_reduce = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_reduce = nn.BatchNorm2d(D*C)

    self.conv_conv = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
    self.bn = nn.BatchNorm2d(D*C)

    self.conv_expand = nn.Conv2d(D*C, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_expand = nn.BatchNorm2d(planes*4)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return F.relu(residual + bottleneck, inplace=True)


class SKLayer(nn.Module):
    def __init__(self, inplanes, stride, sk_groups=24*3) -> None:
        super(SKLayer, self).__init__()
        #planes = max(32, inplanes // reduction_factor)
        self.sk_proj = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 1, groups=sk_groups, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, 1, groups=sk_groups, bias=False),
            nn.BatchNorm2d(inplanes),
        )
        self.sk_downsample = None
        if stride > 1:
          self.sk_downsample = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, stride=stride, padding=1,
                groups=sk_groups, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
          )

    def forward(self, x, fx):
        if self.sk_downsample:
          x = self.sk_downsample(x)
        fuse = F.adaptive_avg_pool2d(x, 1) + F.adaptive_avg_pool2d(fx, 1)
        sk = self.sk_proj(fuse).sigmoid()
        out = (fx - x) * sk + x
        return out

class MSBottleneck(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None,
                 sk_groups = 72, depth = 4): 
        super(MSBottleneck, self).__init__()
        if stride != 1:
          D = int(math.floor(planes * (base_width/64.0)))
          C = cardinality * depth
          depth = 1
        else:
          D = int(math.floor(planes * (base_width/64.0)))
          C = cardinality

        self.conv_reduce = nn.Conv2d(inplanes, D*C*depth, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D*C*depth)
        convs = []
        bns = []
        if depth == 1:
          self.nums = 1
        else:
          self.nums = depth -1 
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.sk = SKLayer(D*C*self.nums, stride, sk_groups=sk_groups)

        self.conv_expand = nn.Conv2d(D*C*depth, planes*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(planes*4)
      
        self.downsample = downsample
        self.width  = D*C
        self.depth = depth

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)
        spx = torch.split(bottleneck, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = self.convs[i](spx[i])
            sp = F.relu(self.bns[i](sp), inplace=True)
            bottleneck = sp
          else:
            sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp), inplace=True)
            bottleneck = torch.cat((bottleneck, sp), 1)

        old_bottleneck = torch.cat(spx[0: self.nums], 1)       
        bottleneck = self.sk(old_bottleneck, bottleneck)

        if self.nums != 1 or self.depth == 2:
          bottleneck = torch.cat((bottleneck,spx[self.nums]),1)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
          residual = self.downsample(x)
        
        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, block, depth, cardinality, base_width, scale, sk_groups):
        super(CifarResNeXt, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.scale = scale
        self.sk_groups = sk_groups

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64 , layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, stride, 
                            downsample, sk_groups=self.sk_groups,
                            depth=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width, 
                            sk_groups=self.sk_groups, depth=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        return x

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
        else:
            raise TypeError('pretrained must be a str or None')


def resnext29_16_64(num_classes=100):
  """Constructs a ResNeXt-29, 16*64d model for CIFAR-100 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 16, 64, num_classes)
  return model

def resnext29_8_64(num_classes=100):
  """Constructs a ResNeXt-29, 8*64d model for CIFAR-100 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(ResNeXtBottleneck, 29, 8, 64, num_classes)
  return model

def resnexts29(num_classes=100):
  """Constructs a ResNeXt-29, 8*64d model for CIFAR-100 (by default)
  
  Args:
    num_classes (uint): number of classes
  """
  model = CifarResNeXt(MSBottleneck, 29, 6, 24, num_classes)
  return model

@BACKBONES.register_module()
class SK2NeXt29(CifarResNeXt):
    def __init__(self, block=MSBottleneck, depth=29, cardinality=6, base_width=24, scale=6, sk_groups=72):
        super().__init__(block, depth, cardinality, base_width, scale, sk_groups)


