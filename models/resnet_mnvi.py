from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import logging
import torch
from contrib import varprop

import torch.nn.functional as F


def keep_variance(x, min_variance):
    return x.clamp(min=min_variance)

def finitialize(modules):
    logging.info("Initializing MSRA")
    for layer in modules:
        if isinstance(layer, (varprop.Conv2dMN, varprop.LinearMN)):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, varprop.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, mnv_init=-3.0, prior_precision=1e0):
    """3x3 convolution with padding"""
    return varprop.Conv2dMN(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation, mnv_init=mnv_init, prior_precision=prior_precision)

def conv1x1(in_planes, out_planes, stride=1, mnv_init=-2.0, prior_precision=1e2):
    """1x1 convolution"""
    return varprop.Conv2dMN(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                    mnv_init=mnv_init, prior_precision=prior_precision)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, keep_variance_fn=None, mnv_init=-3.0, prior_precision=1e0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = varprop.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, mnv_init=mnv_init, prior_precision=prior_precision)
        self.bn1 = norm_layer(planes)
        self.relu = varprop.ReLU(keep_variance_fn=keep_variance_fn)
        self.conv2 = conv3x3(planes, planes, mnv_init=mnv_init, prior_precision=prior_precision)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


    def forward_sampling(self, inputs):
        identity = inputs

        out = self.conv1.forward_sampling(inputs)
        out = self.bn1.forward_sampling(out)
        out = F.relu(out)

        out = self.conv2.forward_sampling(out)
        out = self.bn2.forward_sampling(out)

        if self.downsample is not None:
            identity = self.downsample.forward_sampling(inputs)

        out += identity
        out = F.relu(out)

        return out


    def forward(self, inputs_mean, inputs_variance):
        identity_mean, identity_variance = inputs_mean, inputs_variance

        out = self.conv1(inputs_mean, inputs_variance)
        out = self.bn1(*out)
        out = self.relu(*out)

        out = self.conv2(*out)
        out_mean, out_variance = self.bn2(*out)

        if self.downsample is not None:
            identity_mean, identity_variance = self.downsample(inputs_mean, inputs_variance)

        out_mean += identity_mean
        out_variance += identity_variance
        out_mean, out_variance = self.relu(out_mean, out_variance)

        return out_mean, out_variance


class ResNetMNVI(nn.Module):

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, min_variance=1e-5, mnv_init=-3.0, prior_precision=1e0):
        super(ResNetMNVI, self).__init__()
        if norm_layer is None:
            norm_layer = varprop.BatchNorm2d
        self._norm_layer = norm_layer
        self._keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._mnv_init = mnv_init
        self._prior_precision =  prior_precision

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = varprop.Conv2dMN(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False, mnv_init=self._mnv_init, prior_precision=self._prior_precision)
        self.bn1 = varprop.BatchNorm2d(self.inplanes)
        self.relu = varprop.ReLU(keep_variance_fn=self._keep_variance_fn)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = varprop.AdaptiveAvgPool2d()
        self.fc = varprop.LinearMN(512 * block.expansion, num_classes, mnv_init=self._mnv_init, prior_precision=self._prior_precision)

        finitialize(self.modules())


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = varprop.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, mnv_init=self._mnv_init, prior_precision=self._prior_precision),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            keep_variance_fn=self._keep_variance_fn, mnv_init=self._mnv_init, prior_precision=self._prior_precision))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                keep_variance_fn=self._keep_variance_fn, mnv_init=self._mnv_init, prior_precision=self._prior_precision))

        return varprop.Sequential(*layers)

    def _forward_impl(self, inputs_mean, inputs_variance):
        x = self.conv1(inputs_mean, inputs_variance)
        x = self.bn1(*x)
        x = self.relu(*x)

        x = self.layer1(*x)
        x = self.layer2(*x)
        x = self.layer3(*x)
        x = self.layer4(*x)

        x_mean, x_variance = self.avgpool(*x)
        x_mean = torch.flatten(x_mean, 1)
        x_variance = torch.flatten(x_variance, 1)
        out_mean, out_variance = self.fc(x_mean, x_variance)

        return out_mean, out_variance

    def forward_sampling(self, inputs):
        x = self.conv1.forward_sampling(inputs)
        x = self.bn1.forward_sampling(x)
        x = F.relu(x)

        x = self.layer1.forward_sampling(x)
        x = self.layer2.forward_sampling(x)
        x = self.layer3.forward_sampling(x)
        x = self.layer4.forward_sampling(x)
        
        x = self.avgpool.forward_sampling(x)
        x = torch.flatten(x, 1)
        out = self.fc.forward_sampling(x)

        return out

    def forward(self, inputs_mean, inputs_variance):
        return self._forward_impl(inputs_mean, inputs_variance)

    def kl_div(self):
        kl = 0.0
        for module in self.modules():
            if isinstance(module, (varprop.LinearMN, varprop.Conv2dMN)):
                kl += module.kl_div()
        return kl


class ResNet18MNVI(nn.Module):
    def __init__(self, num_classes=1000, min_variance=1e-5, mnv_init=-3.0, prior_precision=1e0, kl_div_weight=0.0, **kwargs):
        super(ResNet18MNVI, self).__init__()
        self._kl_div_weight = kl_div_weight
        self.resnet = ResNetMNVI(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, min_variance=min_variance,
            mnv_init=mnv_init, prior_precision=prior_precision, **kwargs)

    def forward(self, example_dict):
        inputs = example_dict['input1']
        inputs_mean = inputs
        inputs_variance = torch.zeros_like(inputs_mean)
        prediction_mean, prediction_variance = self.resnet(inputs_mean, inputs_variance)
        return {'prediction_mean': prediction_mean, 'prediction_variance': prediction_variance, 'kl_div': self.kl_div}

    def forward_sampling(self, example_dict):
        inputs = example_dict['input1']
        prediction = self.resnet.forward_sampling(inputs)
        return prediction

    def kl_div(self):
        return self._kl_div_weight * self.resnet.kl_div()