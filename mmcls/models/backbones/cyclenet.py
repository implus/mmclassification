import logging
import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
from mmcv.cnn import (constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES

def conv_bn(in_channels, out_channels, kernel_size, stride, padding=0, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class SinglePath(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, groups=1) -> None:
        super().__init__()
        self.path = conv_bn(inplanes, outplanes, kernel_size, stride, padding, groups)
    
    def forward(self, x):
        return self.path(x)

class MultiPath(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, groups=1) -> None:
        super().__init__()
        self.path_id = nn.BatchNorm2d(inplanes) if inplanes == outplanes and stride == 1 else None
        self.path_3x3 = conv_bn(inplanes, outplanes, kernel_size, stride, padding, groups)
        self.path_1x1 = conv_bn(inplanes, outplanes, 1, stride, padding=0, groups=groups)
    
    def forward(self, x):
        if self.path_id is None:
            id_out = 0
        else:
            id_out = self.path_id(x)
        return self.path_3x3(x) + self.path_1x1(x) + id_out

class CycleConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size,
                 stride=1, padding=0, groups=2, single_path=False) -> None:
        super().__init__()

        self.single_path = single_path
        self.stride = stride
        path_block = SinglePath if single_path else MultiPath

        in_channel = inplanes // 2
        out_channel = outplanes // 2

        self.lr_aa = path_block(in_channel, out_channel, 3, stride, padding=1, groups=groups)
        self.lr_bb = path_block(in_channel, out_channel, 3, stride, padding=1, groups=groups)
        self.lr_ab = path_block(out_channel, out_channel, 3, 1, padding=1, groups=1)

        self.rl_bb = path_block(in_channel, out_channel, 3, stride, padding=1, groups=groups)
        self.rl_aa = path_block(in_channel, out_channel, 3, stride, padding=1, groups=groups)
        self.rl_ba = path_block(out_channel, out_channel, 3, 1, padding=1, groups=1)

        self.relu = nn.ReLU()

    def forward(self, xa, xb):
        if self.stride > 1:
            nxa_0 = self.relu(self.lr_aa(xa))
            nxb_0 = self.relu(self.lr_bb(xb) + self.lr_ab(nxa_0))
            nxb_1 = self.relu(self.rl_bb(xb))
            nxa_1 = self.relu(self.rl_aa(xa) + self.rl_ba(nxb_1))
            xa = nxa_0 + nxa_1
            xb = nxb_0 + nxb_1
        else:
            xa = self.relu(self.lr_aa(xa))
            xb = self.relu(self.lr_bb(xb) + self.lr_ab(xa))
            xb = self.relu(self.rl_bb(xb))
            xa = self.relu(self.rl_aa(xa) + self.rl_ba(xb))
        return xa, xb

class CycleConvSeq(nn.Module):
    def __init__(self, inplanes, planes, num_blocks, stride, single_path) -> None:
        super().__init__()

        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(CycleConvBlock(inplanes=inplanes, outplanes=planes, kernel_size=3,
                stride=stride, groups=2, single_path=single_path))
            inplanes = planes
        self.blocks = nn.ModuleList(blocks)
        self.num_blocks = num_blocks

    def forward(self, xa, xb):
        for i in range(self.num_blocks):
            xa, xb = self.blocks[i](xa, xb)
        return xa, xb

@BACKBONES.register_module()
class CycLeNet(nn.Module):

    def __init__(self, num_blocks, width_multiplier, single_path=False,
                 out_indices=(0,1,2,3), style='pytorch'):
        super().__init__()

        assert len(width_multiplier) == 4

        self.inplanes = min(64, int(64 * width_multiplier[0]))

        self.single_path = single_path
        self.stage0 = CycleConvBlock(6, self.inplanes, 3, 2, 1, groups=1, single_path=single_path)
        planes = int(64 *  width_multiplier[0])
        self.stage1 = CycleConvSeq(self.inplanes, planes, num_blocks[0], stride=2, single_path=single_path)
        self.inplanes = planes

        planes = int(128 *  width_multiplier[1])
        self.stage2 = CycleConvSeq(self.inplanes, planes, num_blocks[1], stride=2, single_path=single_path)
        self.inplanes = planes

        planes = int(256 *  width_multiplier[2])
        self.stage3 = CycleConvSeq(self.inplanes, planes, num_blocks[2], stride=2, single_path=single_path)
        self.inplanes = planes

        planes = int(512 *  width_multiplier[3])
        self.stage4 = CycleConvSeq(self.inplanes, planes, num_blocks[3], stride=2, single_path=single_path)
        self.inplanes = planes
        
    #def _make_stage(self, planes, num_blocks, stride):
    #    strides = [stride] + [1]*(num_blocks-1)
    #    blocks = []
    #    for stride in strides:
    #        blocks.append(CycleConvBlock(inplanes=self.inplanes, outplanes=planes, kernel_size=3,
    #            stride=stride, groups=2, single_path=self.single_path))
    #        self.in_planes = planes
    #    return nn.Sequential(*blocks)

    def forward(self, x):
        xa, xb = self.stage0(x, x)
        xa, xb = self.stage1(xa, xb)
        xa, xb = self.stage2(xa, xb)
        xa, xb = self.stage3(xa, xb)
        xa, xb = self.stage4(xa, xb)
        out = torch.cat((xa, xb), 1)
        return out

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
