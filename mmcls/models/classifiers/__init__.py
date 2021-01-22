from .base import BaseClassifier
from .image import ImageClassifier
from .mixup_image import MixUpImageClassifier
from .cutmix_image import CutMixImageClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'MixUpImageClassifier',
           'CutMixImageClassifier']
