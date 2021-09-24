import os
import os.path
import random

import torch.utils.data as data
import torchvision
from torchvision import transforms

meta = {
    'seed': 58486484
}

class _MNIST(data.Dataset):

    def __init__(self, root, split='train', transform=None,  download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if self.split == 'train':
            self._data = torchvision.datasets.MNIST(root=self.root, train=True, download=download, transform=None)
            random.seed(meta['seed'])
            indices = list(range(0, 60000))
            random.shuffle(indices)
            train_indices = indices[:55000]
            self._data = data.Subset(self._data, train_indices)
        elif self.split == 'valid':
            self._data = torchvision.datasets.MNIST(root=self.root, train=True, download=download, transform=None)
            indices = list(range(0, 60000))
            random.seed(meta['seed'])
            random.shuffle(indices)
            val_indices = indices[55000:]
            self._data = data.Subset(self._data, val_indices)
        elif self.split == 'test':
            self._data = torchvision.datasets.MNIST(root=self.root, train=False, download=download, transform=None)

    def __getitem__(self, index):
        img, target = self._data[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self._data)


class Mnist(data.Dataset):
    def __init__(self, mnist):
        self._mnist = mnist

    def __getitem__(self, index):
        data, target = self._mnist[index]
        example_dict = {
            "input1": data,
            "target1": target,
            "index": index,
            "basename": "img-%05i" % index
        }
        return example_dict

    def __len__(self):
        return len(self._mnist)


class MnistTrain(Mnist):
    def __init__(self, args, root):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist = _MNIST(root, split='train', download=True, transform=transform)
        super(MnistTrain, self).__init__(mnist)


class MnistValid(Mnist):
    def __init__(self, args, root):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist = _MNIST(root, split='valid', download=True, transform=transform)
        super(MnistValid, self).__init__(mnist)

class MnistTest(Mnist):
    def __init__(self, args, root):
        d = os.path.dirname(root)
        if not os.path.exists(d):
            os.makedirs(d)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist = _MNIST(root, split='test', download=True, transform=transform)
        super(MnistTest, self).__init__(mnist)
