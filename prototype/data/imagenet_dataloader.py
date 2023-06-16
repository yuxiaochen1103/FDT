import random
from torchvision import transforms
from .transforms import TwoCropsTransform, CALSMultiResolutionTransform, GaussianBlur, CLSAAug, RandomCropMinSize, SLIPTransform
from .auto_augmentation import ImageNetPolicy
import os
import torch


def build_common_augmentation(aug_type):
    """
    common augmentation settings for training/testing ImageNet
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if aug_type == 'STANDARD':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    if aug_type == 'STANDARD256':
        augmentation = [
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'STANDARD_SLIP':  # same as slip
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'STANDARD_CLIP':  # same as clip
        augmentation = [
            RandomCropMinSize(224),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'AUTOAUG':
        augmentation = [
            transforms.RandomResizedCrop(224),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'MOCOV1':
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type in ['MOCOV2', 'SIMCLR', 'SIMSIAM', 'MOCOV2_single']:
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),  # 0.08-1
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type in ['MOCOV2_256']:
        augmentation = [
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif 'CLSA' in aug_type:
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        augmentation = transforms.Compose(augmentation)

        stronger_aug = CLSAAug(num_of_times=int(aug_type[4])) # num of repetive times for randaug, CLSA5-16-32
        stronger_aug = transforms.Compose([stronger_aug, transforms.ToTensor(), normalize])
    elif aug_type == 'LINEAR':
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]

    elif aug_type == 'ONECROP_nonorm':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    ]

    elif aug_type == 'ONECROP256':
        augmentation = [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'ONECROP384':
        augmentation = [
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == 'SLIP':
        pass
    else:
        raise RuntimeError("undefined augmentation type for ImageNet!")

    if aug_type in ['MOCOV1', 'MOCOV2', 'SIMCLR', 'SIMSIAM', 'MOCOV2_256']:
        return TwoCropsTransform(transforms.Compose(augmentation))
    elif 'CLSA' in aug_type:
        # aug_type = 'CLSA5-16-24-32'
        # aug_type = 'CLSA5-16_32'
        if '_' in aug_type:
            res_range = [int(e) for e in aug_type.split('-')[1].split('_')]
            res = [random.choice(range(res_range[0], res_range[1]+1)), ]
        else:
            res = [int(e) for e in aug_type.split('-')[1:]]
        return CALSMultiResolutionTransform(
            base_transform=augmentation,
            stronger_transfrom=stronger_aug, num_res=len(res),
            resolutions=res)
    elif aug_type == 'SLIP':
        base = build_common_augmentation('STANDARD_SLIP')
        mocov2 = build_common_augmentation('MOCOV2')
        return SLIPTransform(base, mocov2)
    else:
        return transforms.Compose(augmentation)



