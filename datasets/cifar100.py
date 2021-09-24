import os
import os.path
import random

import torch.utils.data as data
import torchvision
from torchvision import transforms as vision_transforms
from . import transforms



# meta data for cifar images and classes
meta = {
    'rgb_mean': (0.5071, 0.4867, 0.4408),
    'rgb_std': (0.2675, 0.2565, 0.2761),
    'seed': 53484894
}




class _CIFAR100(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 distortion=False,
                 photometric_augmentations=None,
                 affine_augmentations=None,
                 random_flip=False,
                 normalize_colors=False,
                 per_image_std=False,
                 download=False,
                 crop=None):

        self.root = os.path.expanduser(root)
        self.split = split
        self.per_image_std = per_image_std

        normalize_colors_transform = transforms.Identity()
        affine_transform = transforms.Identity()
        flip_transform = transforms.Identity()
        noise_transform = transforms.Identity()
        crop_transform = transforms.Identity()

        if crop is not None:
            if train:
                crop_transform = vision_transforms.RandomCrop(crop)
            else:
                crop_transform = vision_transforms.CenterCrop(crop)

        if normalize_colors:
            normalize_colors_transform = vision_transforms.Normalize(
                mean=meta['rgb_mean'], std=meta['rgb_std'])

        self._photometric_transform = transforms.Identity()

        if affine_augmentations is not None:
            affine_transform = vision_transforms.Compose([
                vision_transforms.RandomCrop(32, padding=affine_augmentations['translate']),
                vision_transforms.RandomRotation(affine_augmentations['degrees']),
                ])

        if random_flip:
            flip_transform = vision_transforms.RandomHorizontalFlip()


        if photometric_augmentations is not None:
            brightness_max_delta = photometric_augmentations['brightness_max_delta']
            contrast_max_delta   = photometric_augmentations['contrast_max_delta']
            saturation_max_delta = photometric_augmentations['saturation_max_delta']
            hue_max_delta        = photometric_augmentations['hue_max_delta']
            gamma_min, gamma_max = photometric_augmentations['gamma_min_max']
            self._photometric_transform = vision_transforms.Compose([
                crop_transform,
                vision_transforms.ColorJitter(
                    brightness=brightness_max_delta,
                    contrast=contrast_max_delta,
                    saturation=saturation_max_delta,
                    hue=hue_max_delta),
                affine_transform,
                flip_transform,
                vision_transforms.transforms.ToTensor(),
                transforms.RandomGamma(min_gamma=gamma_min, max_gamma=gamma_max, clip_image=True),
                noise_transform,
                normalize_colors_transform
            ])
        else:
            self._photometric_transform = vision_transforms.Compose([
                crop_transform,
                affine_transform,
                flip_transform,
                vision_transforms.transforms.ToTensor(),
                noise_transform,
                normalize_colors_transform
            ])

        if self.split == 'train':
            self._data = torchvision.datasets.CIFAR100(root=self.root, train=True, download=download, transform=None)
            random.seed(meta['seed'])
            indices = list(range(0, 50000))
            random.shuffle(indices)
            train_indices = indices[:45000]
            self._data = data.Subset(self._data, train_indices)
        elif self.split == 'valid':
            self._data = torchvision.datasets.CIFAR100(root=self.root, train=True, download=download, transform=None)
            random.seed(meta['seed'])
            indices = list(range(0, 50000))
            random.shuffle(indices)
            val_indices = indices[45000:]
            self._data = data.Subset(self._data, val_indices)
        elif self.split == 'test':
            self._data = torchvision.datasets.CIFAR100(root=self.root, train=False, download=download, transform=None)

        

    def __getitem__(self, index):
        img, target = self._data[index]
        img = self._photometric_transform(img)
        return img, target

    def __len__(self):
        return len(self._data)


class Cifar100Base(data.Dataset):
    def __init__(self, cifar):
        super(Cifar100Base, self).__init__()
        self._cifar = cifar

    def __getitem__(self, index):
        data, target = self._cifar[index]
        example_dict = {
            'input1': data,
            'target1': target,
            'index': index,
            'basename': 'img-%05i' % index
        }

        return example_dict

    def __len__(self):
        return len(self._cifar)


class Cifar100Train(Cifar100Base):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations={ 'brightness_max_delta': 0.2,
                                             'contrast_max_delta': 0.2,
                                             'saturation_max_delta': 0.2,
                                             'hue_max_delta': 0.1,
                                             'gamma_min_max': [0.9, 1.1] },
                 affine_augmentations={ 'degrees': 15,
                                        'translate': 4},
                 random_flip=True,
                 normalize_colors=True,
                 per_image_std=False,
                 crop=None):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        cifar = _CIFAR100(
            root,
            split='train',
            download=True,
            crop=crop,
            photometric_augmentations=photometric_augmentations,
            affine_augmentations=affine_augmentations,
            random_flip=random_flip,
            normalize_colors=normalize_colors,
            per_image_std=per_image_std)
        super(Cifar100Train, self).__init__(cifar)


class Cifar100Valid(Cifar100Base):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=None,
                 affine_augmentations=None,
                 random_flip=False,
                 crop=None,
                 normalize_colors=True):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        cifar = _CIFAR100(
            root,
            split='valid',
            download=True,
            crop=crop,
            photometric_augmentations=photometric_augmentations,
            affine_augmentations=affine_augmentations,
            random_flip=random_flip,
            normalize_colors=normalize_colors)
        super(Cifar100Valid, self).__init__(cifar)


class Cifar100Test(Cifar100Base):
    def __init__(self,
                 args,
                 root,
                 photometric_augmentations=None,
                 affine_augmentations=None,
                 random_flip=False,
                 crop=None,
                 normalize_colors=True):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        cifar = _CIFAR100(
            root,
            split='test',
            download=True,
            crop=crop,
            photometric_augmentations=photometric_augmentations,
            affine_augmentations=affine_augmentations,
            random_flip=random_flip,
            normalize_colors=normalize_colors)
        super(Cifar100Test, self).__init__(cifar)
