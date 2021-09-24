from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
import torch
from contrib import varprop


def keep_variance(x, min_variance):
    return x.clamp(min=min_variance)


def finitialize(modules):
    logging.info("Initializing Xavier")
    for layer in modules:
        if isinstance(layer, varprop.Conv2dMFS):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def make_conv(inchannels, outchannels, kernel_size, stride, nonlinear=True, keep_variance_fn=None, prior_precision=1e2):
    padding = kernel_size // 2
    if nonlinear:
        return varprop.Sequential(
            varprop.Conv2dMFS(
                inchannels, outchannels, kernel_size=kernel_size, padding=padding,
                stride=stride, bias=True, prior_precision=prior_precision),
            varprop.ReLU(keep_variance_fn=keep_variance_fn)
        )
    else:
        return varprop.Conv2dMFS(
            inchannels, outchannels, kernel_size=kernel_size, padding=padding, stride=stride, bias=True, prior_precision=prior_precision)



class AllConvNetMFSVI(nn.Module):
    def __init__(self, num_classes=10, min_variance=1e-5, prior_precision=1e0, kl_div_weight=0.0):
        super(AllConvNetMFSVI, self).__init__()
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._prior_precision =  prior_precision
        self._kl_div_weight = kl_div_weight
        self._num_classes = num_classes

        self.conv1   = make_conv(3, 96, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv1_1 = make_conv(96, 96, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv1_2 = make_conv(96, 96, kernel_size=3, stride=2, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)

        self.conv2   = make_conv( 96, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv2_1 = make_conv(192, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv2_2 = make_conv(192, 192, kernel_size=3, stride=2, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)

        self.conv3   = make_conv(192, 192, kernel_size=3, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv3_1 = make_conv(192, 192, kernel_size=1, stride=1, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)
        self.conv3_2 = make_conv(192,  self._num_classes, kernel_size=1, stride=1, nonlinear=False, keep_variance_fn=self._keep_variance_fn, prior_precision=self._prior_precision)

        finitialize(self.modules())

    def forward(self, example_dict):
        inputs = example_dict['input1']
        inputs_mean = inputs
        inputs_variance = torch.zeros_like(inputs_mean)
        x = inputs_mean, inputs_variance

        x = self.conv1(*x)
        x = self.conv1_1(*x)
        x = self.conv1_2(*x)

        x = self.conv2(*x)
        x = self.conv2_1(*x)
        x = self.conv2_2(*x)

        x = self.conv3(*x)
        x = self.conv3_1(*x)
        x = self.conv3_2(*x)
        mu, variance = x

        batch_size = mu.size(0)

        mu = mu.contiguous().view(batch_size, self._num_classes, -1)
        variance = variance.contiguous().view(batch_size, self._num_classes, -1)

        N = mu.size(2)
        mu = mu.mean(dim=2)
        variance = variance.mean(dim=2) * (1.0 / float(N))

        return {'prediction_mean': mu, 'prediction_variance': variance, 'kl_div': self.kl_div}

    def kl_div(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, (varprop.LinearMFS, varprop.Conv2dMFS)):
                kl += module.kl_div()
        return self._kl_div_weight * kl
