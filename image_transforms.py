import random

import torch
from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


# taken from DINO repository
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_size, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        if local_crops_number > 0:
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(p=0.5),
                normalize,
            ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

