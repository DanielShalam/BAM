import torch
from PIL.Image import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

import image_transforms
from contrastive_croping.contrastive_crop import ContrastiveCrop


class DataAugmentationCCrop(object):
    def __init__(self, global_crops_scale, local_crops_scale=None, local_crops_size=None, local_crops_number=None,
                 alpha: float = 0.4):

        # Normalize -

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Horizontal flip, Grey-scale & Color jitter -

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        # Global crop 1 -

        self.global_transfo1 = transforms.Compose([

            ContrastiveCrop(
                alpha=alpha, size=224, scale=(global_crops_scale[0], 1.),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),

            flip_and_color_jitter,
            image_transforms.GaussianBlur(1.),
            normalize
        ])

        # Global crop 2 -

        self.global_transfo2 = transforms.Compose([

            ContrastiveCrop(
                alpha=alpha, size=224, scale=(global_crops_scale[0], 1.),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),

            flip_and_color_jitter,
            image_transforms.GaussianBlur(0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            normalize
        ])

        # Local crop -

        self.local_crops_number = local_crops_number

        if local_crops_number > 0:
            self.local_transfo = transforms.Compose([
                ContrastiveCrop(
                    alpha=alpha, size=local_crops_size, scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                flip_and_color_jitter,
                image_transforms.GaussianBlur(p=0.5),
                normalize,
            ])

    def __call__(self, image):

        crops = [self.global_transfo1(image), self.global_transfo2(image)]

        if self.local_crops_number > 0:
            crops = crops + [self.local_transfo(image) for _ in range(self.local_crops_number)]

        return crops


class DataAugmentationCCropSameHFlip(object):
    def __init__(self, global_crops_scale, local_crops_scale=None, local_crops_size=None, local_crops_number=None,
                 alpha: float = 0.4):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.global_transfo1 = transforms.Compose([
            ContrastiveCrop(alpha=alpha, size=224, scale=(global_crops_scale[0], 1.)),
            color_jitter,
            image_transforms.GaussianBlur(1.),
            normalize
        ])

        self.global_transfo2 = transforms.Compose([
            ContrastiveCrop(alpha=alpha, size=224, scale=(global_crops_scale[0], 1.)),
            color_jitter,
            image_transforms.GaussianBlur(0.1),
            image_transforms.Solarization(0.2),
            normalize
        ])

        self.local_crops_number = local_crops_number
        if local_crops_number > 0:
            self.local_transfo = transforms.Compose([
                ContrastiveCrop(alpha=alpha, size=local_crops_size, scale=local_crops_scale,
                                interpolation=InterpolationMode.BICUBIC),
                color_jitter,
                image_transforms.GaussianBlur(p=0.5),
                normalize,
            ])
        self.random_hflip = transforms.RandomHorizontalFlip()

    def __call__(self, image):
        # we first flip the image for all augmentations
        image = self.random_hflip(image)
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class ImageFolderCCrop(ImageFolder):

    def __init__(self, root, transform_rcrop, transform_ccrop, init_box=None, **kwargs):
        super().__init__(root=root, **kwargs)
        # transform
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop

        init_box = init_box or [0., 0., 1., 1.]
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_box:
            box = self.boxes[index].float().tolist()  # box=[h_min, w_min, h_max, w_max]
            sample = self.transform_ccrop([sample, box])
        else:
            sample = self.transform_rcrop(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
