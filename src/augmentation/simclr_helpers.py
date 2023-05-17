# https://github.com/AndrewAtanov/simclr-pytorch/blob/master/models/ssl.py

import torch
from torchvision import transforms
import numpy as np

import PIL
import torchvision.transforms.functional as FT
from PIL import Image
from pdb import set_trace as pb
# from .py_albument import ColorJitterALB

class GaussianBlur(object):
    """
        PyTorch version of
        https://github.com/google-research/simclr/blob/244e7128004c5fd3c7805cf3135c79baa6c3bb96/data_util.py#L311
    """
    def gaussian_blur(self, image, sigma):
        image = image.reshape(1, 3, 224, 224)
        radius = int(self.kernel_size/2)
        kernel_size = radius * 2 + 1
        x = np.arange(-radius, radius + 1)

        blur_filter = np.exp(
              -np.power(x, 2.0) / (2.0 * np.power(float(sigma), 2.0)))
        blur_filter /= np.sum(blur_filter)

        conv1 = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), groups=3, padding=[kernel_size//2, 0], bias=False)
        conv1.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 0, 1])))

        conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), groups=3, padding=[0, kernel_size//2], bias=False)
        conv2.weight = torch.nn.Parameter(
            torch.Tensor(np.tile(blur_filter.reshape(kernel_size, 1, 1, 1), 3).transpose([3, 2, 1, 0])))

        res = conv2(conv1(image))
        assert res.shape == image.shape
        return res[0]

    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            if np.random.uniform() < self.p:
                return self.gaussian_blur(img, sigma=np.random.uniform(0.2, 2))
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0}, p={1})'.format(self.kernel_size, self.p)


class Clip(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1)

# https://github.com/AndrewAtanov/simclr-pytorch/blob/d147c6bea1787e0d68dd334327c93e69b56f601e/utils/datautils.py#L77
class CenterCropAndResize(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, proportion, size):
        self.proportion = proportion
        self.size = size
        
        smallest_dim = int(round(self.size/self.proportion))

        self.transform = transforms.Compose([
                transforms.Resize(size=smallest_dim, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=self.size),
            ])

    def __call__(self, img):
        return self.transform(img)

    def __repr__(self):
        return self.__class__.__name__ + '(proportion={0}, size={1})'.format(self.proportion, self.size)

# ========================================================

class ConditionalResize(object):
    def __init__(self, size):
        self.size = size

        self.transform = transforms.Compose([
                transforms.Resize(size=self.size, interpolation=transforms.InterpolationMode.BICUBIC)
            ])

    def __call__(self, img):
        img_size = FT.get_image_size(img)
        if (img_size[0] != self.size or img_size[1] != self.size):
            return self.transform(img)
        else:
            return img


# ========================================================


def get_color_distortion(s=1.0, source='pytorch'):
    # s is the strength of color distortion.
    # given from https://arxiv.org/pdf/2002.05709.pdf
    
    # if source =='albumentations':
    #     color_jitter = ColorJitterALB(s)
    if source =='pytorch':
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray
    ])
    return color_distort
