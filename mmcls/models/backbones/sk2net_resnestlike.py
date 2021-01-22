import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet

class SKLayer(nn.Module):
    def __init__(self, inplanes, reduction_factor=4) -> None:
        super(SKLayer, self).__init__()
        planes = max(32, inplanes // reduction_factor)
        self.sk_proj = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, inplanes, 1, bias=False),
            nn.BatchNorm2d(inplanes),
        )

    def forward(self, x, fx):
        fuse = F.adaptive_avg_pool2d(x, 1) + F.adaptive_avg_pool2d(fx, 1)
        sk = self.sk_proj(fuse).sigmoid()
        out = (fx - x) * sk + x
        return out

class Bottle2neck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 scales=4,
                 base_width=32,
                 base_channels=64,
                 cardinality=8,
                 reduction_factor=4,
                 down_stride_after=True, 
                 stage_type='normal',
                 **kwargs):
        """Bottle2neck block for SK2Net.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottle2neck, self).__init__(inplanes, planes, **kwargs)
        assert scales > 1, 'SK2Net degenerates to ResNet when scales = 1.'
        width = int(math.floor(self.planes * (base_width / base_channels)))
        self.down_stride_after = down_stride_after

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.pool = None
        if stage_type == 'stage' and self.conv2_stride != 1:
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=self.conv2_stride, padding=1)
        convs = []
        bns = []

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            for i in range(scales - 1):
                convs.append(
                    build_conv_layer(
                        self.conv_cfg,
                        width,
                        width,
                        kernel_size=3,
                        groups=cardinality,
                        stride=1, # always 1, not self.conv2_stride
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            for i in range(scales - 1):
                convs.append(
                    build_conv_layer(
                        self.dcn,
                        width,
                        width,
                        kernel_size=3,
                        groups=cardinality,
                        stride=1, # always 1, not self.conv2_stride
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False))
                bns.append(
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])
            self.convs = nn.ModuleList(convs)
            self.bns = nn.ModuleList(bns)

        self.sk = SKLayer(width * (scales - 1), reduction_factor)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            if not self.down_stride_after and self.pool:
                out = self.pool(out)

            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                if self.stage_type == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            old_out = torch.cat(spx[:self.scales - 1], 1)
            out = self.sk(old_out, out)

            out = torch.cat((out, spx[self.scales - 1]), 1)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            if self.down_stride_after and self.pool:
                out = self.pool(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class SK2Layer(nn.Sequential):
    """SK2Layer to build SK2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in SK2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 32
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=32,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False),
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                base_width=base_width,
                stage_type='stage',
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    **kwargs))
        super(SK2Layer, self).__init__(*layers)


@BACKBONES.register_module()
class SK2Net(ResNet):
    """SK2Net backbone.

    Args:
        scales (int): Scales used in SK2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 32
        reduction_factor (int): divide reduction_factor of SK attention. Default: 13
        depth (int): Depth of res2net, from {50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): SK2net stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import SK2Net
        >>> import torch
        >>> self = SK2Net(depth=50, scales=4, base_width=32)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3))
    }

    def __init__(self,
                 scales=4,
                 base_width=32,
                 cardinality=8,
                 reduction_factor=4,
                 down_stride_after=True,
                 style='pytorch',
                 deep_stem=True,
                 avg_down=True,
                 **kwargs):
        self.scales = scales
        self.base_width = base_width
        self.cardinality = cardinality
        self.reduction_factor = reduction_factor
        self.down_stride_after = down_stride_after
        super(SK2Net, self).__init__(
            style=style, deep_stem=deep_stem, avg_down=avg_down, **kwargs)

    def make_res_layer(self, **kwargs):
        return SK2Layer(
            scales=self.scales,
            base_width=self.base_width,
            base_channels=self.base_channels,
            cardinality=self.cardinality,
            reduction_factor=self.reduction_factor,
            down_stride_after=self.down_stride_after,
            **kwargs)

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

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottle2neck):
                        # dcn in SK2Net bottle2neck is in ModuleList
                        for n in m.convs:
                            if hasattr(n, 'conv_offset'):
                                constant_init(n.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottle2neck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')
