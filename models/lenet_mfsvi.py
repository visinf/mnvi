from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
from contrib import varprop
import torch


def keep_variance(x, min_variance):
    return x.clamp(min=min_variance)

def finitialize(modules):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, (varprop.Conv2dMFS, varprop.LinearMFS)):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


class LeNetMFSVI(nn.Module):
    def __init__(self, num_classes=10, min_variance=1e-5, prior_precision=1e0, kl_div_weight=0.0):
        super(LeNetMFSVI, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._kl_div_weight = kl_div_weight
        self._num_classes = num_classes

        self.conv1 = varprop.Conv2dMFS(1, 32, kernel_size=5, prior_precision=prior_precision)
        self.relu1 = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.maxpool1 = varprop.MaxPool2d(keep_variance_fn=self._keep_variance_fn)
        self.conv2 = varprop.Conv2dMFS(32, 64, kernel_size=5, prior_precision=prior_precision)
        self.relu2 = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.maxpool2 = varprop.MaxPool2d(keep_variance_fn=self._keep_variance_fn)
        self.fc1 = varprop.LinearMFS(1024, 1024, prior_precision=prior_precision)
        self.relu3 = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.fc2 = varprop.LinearMFS(1024, self._num_classes, prior_precision=prior_precision)

        finitialize(self.modules())

    def forward(self, example_dict):
        inputs = example_dict['input1']
        inputs_mean = inputs
        inputs_variance = torch.zeros_like(inputs_mean)
        x = inputs_mean, inputs_variance
        x = self.conv1(*x)
        x = self.relu1(*x)
        x = self.maxpool1(*x)
        x = self.conv2(*x)
        x = self.relu2(*x)
        x = self.maxpool2(*x)
        x = [u.view(-1, 1024) for u in x]
        x = self.fc1(*x)
        x = self.relu3(*x)
        mean, variance = self.fc2(*x)
        return {'prediction_mean': mean, 'prediction_variance': variance, 'kl_div': self.kl_div}

    def kl_div(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, (varprop.LinearMFS, varprop.Conv2dMFS)):
                kl += module.kl_div()
        return self._kl_div_weight * kl

