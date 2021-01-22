import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 dropout_ratio=0.2,
                 topk=(1, )):
        super(LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.in_channels, self.num_classes)
        )

    def init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)

    def simple_test(self, img):
        """Test without augmentation."""
        cls_score = self.fc(img)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        if torch.onnx.is_in_onnx_export():
            return pred
        pred = list(pred.detach().cpu().numpy())
        return pred

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        # compute accuracy
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def mixloss(self, cls_score, gt_a, gt_b, lam):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss_a = self.compute_loss(cls_score, gt_a, avg_factor=num_samples)
        loss_b = self.compute_loss(cls_score, gt_b, avg_factor=num_samples)
        loss = lam * loss_a + (1 - lam) * loss_b
        # compute accuracy
        acc_a = self.compute_accuracy(cls_score, gt_a)
        acc_b = self.compute_accuracy(cls_score, gt_b)
        assert len(acc_a) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': lam*a+(1-lam)*b for k, a, b in zip(self.topk, acc_a, acc_b)}
        return losses

    def forward_train(self, x, gt_label):
        cls_score = self.fc(x)
        if not isinstance(gt_label, tuple):
            losses = self.loss(cls_score, gt_label)
        else:
            gt_a, gt_b, lam = gt_label
            losses = self.mixloss(cls_score, gt_a, gt_b, lam)
        return losses
