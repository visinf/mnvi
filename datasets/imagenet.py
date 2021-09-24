from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

import torch.utils.data as data
from torchvision import transforms
from torchvision import datasets



# meta data for cifar images and classes
meta = {'rgb_mean':[0.485, 0.456, 0.406], 'rgb_std': [0.229, 0.224, 0.225],}






class ImageNetBase(data.Dataset):
    def __init__(self, inet):
        super(ImageNetBase, self).__init__()
        self._inet = inet

    def __getitem__(self, index):
        data, target = self._inet[index]
        example_dict = {
            "input1": data,
            "target1": target,
            "index": index,
        }

        return example_dict

    def __len__(self):
        return len(self._inet)


class ImageNetTrain(ImageNetBase):
    def __init__(self, args, root):
        d = os.path.dirname(root)
        inet = datasets.ImageFolder(
        root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=meta['rgb_mean'], std=meta['rgb_std']),
        ]))
        super(ImageNetTrain, self).__init__(inet)


class ImageNetValid(ImageNetBase):
    def __init__(self, args, root):
        d = os.path.dirname(root)
        inet = datasets.ImageFolder(
        root,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=meta['rgb_mean'], std=meta['rgb_std']),
        ]))
        super(ImageNetValid, self).__init__(inet)
