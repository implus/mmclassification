from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

__all__ = ["ImageNetPolicy"]


class AutoSubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        def func_shearX(img, magnitude):
            return img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                        Image.BICUBIC, fillcolor=fillcolor)

        def func_shearY(img, magnitude):
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                        Image.BICUBIC, fillcolor=fillcolor)

        def func_translateX(img, magnitude):
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                        fillcolor=fillcolor)

        def func_translateY(img, magnitude):
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                        fillcolor=fillcolor)

        def func_rotate(img, magnitude):
            return rotate_with_fill(img, magnitude)

        def func_color(img, magnitude):
            return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))

        def func_posterize(img, magnitude): 
            return ImageOps.posterize(img, magnitude)

        def func_solarize(img, magnitude): 
            return ImageOps.solarize(img, magnitude)

        def func_contrast(img, magnitude):
            return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))

        def func_sharpness(img, magnitude):
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))

        def func_brightness(img, magnitude):
            return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))

        def func_autocontrast(img, magnitude):
            return ImageOps.autocontrast(img)

        def func_equalize(img, magnitude):
            return ImageOps.equalize(img)

        def func_invert(img, magnitude):
            return ImageOps.invert(img)

        func = {
            "shearX": func_shearX,
            "shearY": func_shearY,
            "translateX": func_translateX,
            "translateY": func_translateY,
            "rotate": func_rotate,
            "color":  func_color,
            "posterize": func_posterize,
            "solarize": func_solarize,
            "contrast": func_contrast,
            "sharpness": func_sharpness,
            "brightness": func_brightness,
            "autocontrast": func_autocontrast,
            "equalize": func_equalize,
            "invert": func_invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            AutoSubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            AutoSubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            AutoSubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            AutoSubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            AutoSubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            AutoSubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            AutoSubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            AutoSubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            AutoSubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            AutoSubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            AutoSubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            AutoSubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            AutoSubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            AutoSubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            AutoSubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            AutoSubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            AutoSubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            AutoSubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            AutoSubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            AutoSubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            AutoSubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            AutoSubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            AutoSubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            AutoSubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            AutoSubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            AutoSubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            AutoSubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            AutoSubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            AutoSubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            AutoSubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            AutoSubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            AutoSubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            AutoSubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            AutoSubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            AutoSubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            AutoSubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            AutoSubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            AutoSubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            AutoSubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            AutoSubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            AutoSubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            AutoSubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            AutoSubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            AutoSubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            AutoSubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            AutoSubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            AutoSubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            AutoSubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            AutoSubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            AutoSubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            AutoSubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            AutoSubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            AutoSubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            AutoSubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            AutoSubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            AutoSubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            AutoSubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            AutoSubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            AutoSubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            AutoSubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            AutoSubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            AutoSubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            AutoSubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            AutoSubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            AutoSubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            AutoSubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            AutoSubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            AutoSubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            AutoSubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            AutoSubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            AutoSubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            AutoSubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            AutoSubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            AutoSubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            AutoSubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"

