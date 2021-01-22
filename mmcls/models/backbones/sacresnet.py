import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvAWS2d, constant_init
from mmcv.utils import TORCH_VERSION

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResLayer, ResNetV1d, ResNet

class SACConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(SACConv2d, self).__init__()
        self.switch = nn.Conv2d(
            in_channels, 1, kernel_size=1, stride=stride, bias=True)
        #self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        assert kernel_size == 3
        self.conv_s = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride, 
            bias=True, padding=1, dilation=1)
        self.conv_l = nn.Conv2d(
            in_channels, channels, kernel_size=3, stride=stride,
            bias=True, padding=3, dilation=3)
        self.init_weights()

    def init_weights(self):
        constant_init(self.switch, 0, bias=1)

    def forward(self, x):
        # switch
        avg_x = F.pad(x, pad=(2, 2, 2, 2), mode='reflect')
        avg_x = F.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # sac
        out_s = self.conv_s(x)
        out_l = self.conv_l(x)
        out = switch * out_s + (1 - switch) * out_l
        return out

class Bottleneck(_Bottleneck):
    """Bottleneck block for ResNeSt.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        groups (int): Groups of conv2.
        width_per_group (int): Width per group of conv2. 64x4d indicates
            ``groups=64, width_per_group=4`` and 32x8d indicates
            ``groups=32, width_per_group=8``.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 #in_channels,
                 #out_channels,
                 inplanes,
                 planes,
                 groups=1,
                 width_per_group=4,
                 base_channels=64,
                 avg_down_stride=True,
                 avg_down_after=True,
                 **kwargs):
        in_channels = inplanes
        out_channels = planes * self.expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = planes
        super(Bottleneck, self).__init__(in_channels, out_channels, **kwargs)

        self.groups = groups
        self.width_per_group = width_per_group

        # For ResNet bottleneck, middle channels are determined by expansion
        # and out_channels, but for ResNeXt bottleneck, it is determined by
        # groups and width_per_group and the stage it is located in.
        if groups != 1:
            assert self.mid_channels % base_channels == 0
            self.mid_channels = (
                groups * width_per_group * self.mid_channels // base_channels)

        self.avg_down_stride = avg_down_stride and self.conv2_stride > 1
        self.avg_down_after = avg_down_after

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, self.mid_channels, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = SACConv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=1, # if self.avg_down_stride else self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)
        delattr(self, self.norm2_name)

        if self.avg_down_stride:
            self.avd_layer = nn.AvgPool2d(3, self.conv2_stride, padding=1)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.mid_channels,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.avg_down_stride and not self.avg_down_after:
                out = self.avd_layer(out)

            out = self.conv2(out)

            if self.avg_down_stride and self.avg_down_after:
                out = self.avd_layer(out)

            out = self.conv3(out)
            out = self.norm3(out)

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


@BACKBONES.register_module()
class SACResNet(ResNet):
    """ResNeSt backbone.

    Please refer to the `paper <https://arxiv.org/pdf/2004.08955.pdf>`_ for
    details.

    Args:
        depth (int): Network depth, from {50, 101, 152, 200}.
        groups (int): Groups of conv2 in Bottleneck. Default: 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Default: 4.
        avg_down_stride (bool): Whether to use average pool for stride in
            Bottleneck. Default: True.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): The config dict for conv layers. Default: None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3)),
        269: (Bottleneck, (3, 30, 48, 8))
    }

    def __init__(self,
                 depth,
                 groups=1,
                 width_per_group=4,
                 avg_down_stride=True, # always true
                 avg_down_after=True,
                 deep_stem=True,
                 avg_down=True,
                 **kwargs):
        self.groups = groups
        self.width_per_group = width_per_group
        self.avg_down_stride = avg_down_stride
        self.avg_down_after = avg_down_after
        super(SACResNet, self).__init__(depth=depth, 
            deep_stem=deep_stem, avg_down=avg_down, **kwargs)

    def make_res_layer(self, **kwargs):
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            avg_down_stride=self.avg_down_stride,
            avg_down_after=self.avg_down_after,
            **kwargs)
