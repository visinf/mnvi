import operator
from collections import OrderedDict
from itertools import islice
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from contrib.math import normpdf, normcdf



class ReLU(nn.Module):
    def __init__(self, keep_variance_fn=None):
        super(ReLU, self).__init__()
        self._keep_variance_fn = keep_variance_fn

    def forward(self, features_mean, features_variance):
        if self._keep_variance_fn is not None:
            features_variance = self._keep_variance_fn(features_variance)
        features_stddev = torch.sqrt(features_variance)
        div = (features_mean / features_stddev)
        pdf = normpdf(div)
        cdf = normcdf(div)
        outputs_mean = features_mean * cdf + features_stddev * pdf
        outputs_variance = (features_mean ** 2 + features_variance) * cdf \
                           + features_mean * features_stddev * pdf - outputs_mean ** 2
        return outputs_mean, outputs_variance


class LinearMN(nn.Module):
    def __init__(self, in_features, out_features, prior_precision=1e0, mnv_init=-3.0, bias=True, eps=1e-10):
        super(LinearMN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision =  prior_precision
        self.eps = eps
        self.has_bias = bias

        self.mult_noise_variance = Parameter(torch.ones(in_features) * mnv_init)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if self.has_bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear((1 + F.softplus(self.mult_noise_variance)) * inputs_variance \
            + F.softplus(self.mult_noise_variance) * inputs_mean**2, self.weight**2)
        return outputs_mean, outputs_variance

    def forward_sampling(self, inputs):
        outputs_mean = F.linear(inputs, self.weight, self.bias)
        outputs_variance = F.linear(F.softplus(self.mult_noise_variance) * inputs**2, self.weight**2)
        normal_dist = torch.distributions.normal.Normal(torch.zeros_like(outputs_mean), torch.ones_like(outputs_mean))
        normals = normal_dist.sample()
        outputs_sample = outputs_mean + torch.sqrt(outputs_variance) * normals
        return outputs_sample

    def kl_div(self):
        kld = 0.5 * (-torch.log(F.softplus(self.mult_noise_variance) * self.weight**2 + self.eps) \
            + self.prior_precision * ((1 + F.softplus(self.mult_noise_variance)) * self.weight**2)).sum()
        return kld



class LinearMF(nn.Module):
    def __init__(self, in_features, out_features, prior_precision=1e0, bias=True, eps=1e-10):
        super(LinearMF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias
        log_var_init =  math.log(1 / prior_precision)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_variance = Parameter(torch.ones_like(self.weight) * log_var_init)
        if self.has_bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_variance = Parameter(torch.ones(out_features) * log_var_init)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_mean**2, torch.exp(self.weight_variance))
        if inputs_variance is not None:
            outputs_variance += F.linear(inputs_variance, self.weight**2 + torch.exp(self.weight_variance))
        if self.has_bias:
            outputs_variance += torch.exp(self.bias_variance)
        return outputs_mean, outputs_variance

    def kl_div(self):
        kld = 0.5 * (-self.weight_variance + self.prior_precision * (torch.exp(self.weight_variance) + self.weight**2)).sum()
        if self.has_bias:
            kld += 0.5 * (-self.bias_variance + self.prior_precision * (torch.exp(self.bias_variance) + self.bias**2)).sum()
        return kld

class LinearMFS(nn.Module):
    def __init__(self, in_features, out_features, prior_precision=1e0, bias=True, eps=1e-10):
        super(LinearMFS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias
        log_var_init =  math.log(1 / prior_precision)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_variance = Parameter(torch.ones(in_features) * log_var_init)
        if self.has_bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_variance = Parameter(torch.ones(out_features) * log_var_init)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs_mean, inputs_variance):
        weight_variance = self.weight_variance.unsqueeze(dim=0).repeat(self.out_features,1)
        outputs_mean = F.linear(inputs_mean, self.weight, self.bias)
        outputs_variance = F.linear(inputs_mean**2 * torch.exp(self.weight_variance), torch.ones_like(self.weight).to(self.weight_variance.device))
        if inputs_variance is not None:
            outputs_variance += F.linear(inputs_variance, self.weight**2 + torch.exp(weight_variance))
        if self.has_bias:
            outputs_variance += torch.exp(self.bias_variance)
        return outputs_mean, outputs_variance

    def kl_div(self):
        weight_variance = self.weight_variance.unsqueeze(dim=0).repeat(self.out_features,1)
        kld = 0.5 * (-weight_variance + self.prior_precision * (torch.exp(weight_variance) + self.weight**2)).sum()
        if self.has_bias:
            kld += 0.5 * (-self.bias_variance + self.prior_precision * (torch.exp(self.bias_variance) + self.bias**2)).sum()
        return kld


class Conv2dMN(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, prior_precision=1e0, mnv_init=-3.0, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-10):
        self.out_channels = out_channels
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMN, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self.mult_noise_variance = Parameter(torch.ones(1, in_channels, 1, 1) * mnv_init)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(
            (1 + F.softplus(self.mult_noise_variance)) * inputs_variance
            + F.softplus(self.mult_noise_variance) * inputs_mean**2,
            self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        return outputs_mean, outputs_variance

    def forward_sampling(self, inputs):
        outputs_mean = F.conv2d(
            inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(
            F.softplus(self.mult_noise_variance) * inputs**2,
            self.weight ** 2, None, self.stride, self.padding, self.dilation, self.groups)
        normal_dist = torch.distributions.normal.Normal(torch.zeros_like(outputs_mean), torch.ones_like(outputs_mean))
        normals = normal_dist.sample()
        outputs_sample = outputs_mean + torch.sqrt(outputs_variance) * normals
        return outputs_sample

    def kl_div(self):
        kld = 0.5 * (-torch.log(F.softplus(self.mult_noise_variance) * self.weight**2 + self.eps) \
            + self.prior_precision * ((1 + F.softplus(self.mult_noise_variance)) * self.weight**2)).sum()
        return kld


class Conv2dMF(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, prior_precision=1e0, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-10):
        self.out_channels = out_channels
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias
        log_var_init =  math.log(1 / prior_precision)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self.weight_variance = Parameter(torch.ones_like(self.weight) * log_var_init)
        if self.has_bias:
            self.bias_variance = Parameter(torch.ones_like(self.bias) * log_var_init)

    def forward(self, inputs_mean, inputs_variance):
        outputs_mean = F.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(inputs_mean**2,
            torch.exp(self.weight_variance), None, self.stride, self.padding, self.dilation, self.groups)
        if inputs_variance is not None:
            outputs_variance += F.conv2d(inputs_variance, torch.exp(self.weight_variance) + self.weight**2, None, self.stride, self.padding, self.dilation, self.groups)
        if self.has_bias:
            outputs_variance += torch.exp(self.bias_variance).view(1, -1, 1, 1)
        return outputs_mean, outputs_variance

    def kl_div(self):
        kld = 0.5 * (-self.weight_variance + self.prior_precision * (torch.exp(self.weight_variance) + self.weight**2)).sum()
        if self.has_bias:
            kld +=  0.5 * (-self.bias_variance + self.prior_precision * (torch.exp(self.bias_variance) + self.bias**2)).sum()
        return kld


class Conv2dMFS(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, prior_precision=1e0, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-10):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.prior_precision = prior_precision
        self.eps = eps
        self.has_bias = bias
        log_var_init =  math.log(1 / prior_precision)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMFS, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, 'zeros')

        self.weight_variance = Parameter(torch.ones(in_channels // groups) * log_var_init)
        if self.has_bias:
            self.bias_variance = Parameter(torch.ones_like(self.bias) * log_var_init)

    def forward(self, inputs_mean, inputs_variance):
        weight_variance = self.weight_variance.view(1, -1, 1, 1).repeat(self.out_channels,1,self.kernel_size[0],self.kernel_size[1])
        outputs_mean = F.conv2d(
            inputs_mean, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputs_variance = F.conv2d(inputs_mean**2,
            torch.exp(weight_variance), None, self.stride, self.padding, self.dilation, self.groups)
        if inputs_variance is not None:
            outputs_variance += F.conv2d(inputs_variance, torch.exp(weight_variance) + self.weight**2, None, self.stride, self.padding, self.dilation, self.groups)
        if self.has_bias:
            outputs_variance += torch.exp(self.bias_variance).view(1, -1, 1, 1)
        return outputs_mean, outputs_variance

    def kl_div(self):
        weight_variance = self.weight_variance.view(1, -1, 1, 1).repeat(self.out_channels,1,self.kernel_size[0],self.kernel_size[1])
        kld = 0.5 * (-weight_variance + self.prior_precision * (torch.exp(weight_variance) + self.weight**2)).sum()
        if self.has_bias:
            kld +=  0.5 * (-self.bias_variance + self.prior_precision * (torch.exp(self.bias_variance) + self.bias**2)).sum()
        return kld


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs_mean, inputs_variance):
        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        outputs_mean = F.batch_norm(
            inputs_mean, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.training:
            batch_variances = inputs_mean.detach().var(dim=(0,2,3)).view(1, -1, 1, 1)
        else:
            batch_variances = self.running_var.view(1, -1, 1, 1)
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            outputs_variance = inputs_variance * weight**2 / (batch_variances + self.eps)
        else:
            outputs_variance = inputs_variance / (batch_variances + self.eps)
        return outputs_mean, outputs_variance

    def forward_sampling(self, inputs):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        outputs = F.batch_norm(
            inputs, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        return outputs



class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, keep_variance_fn=None):
        super(MaxPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, return_indices=True)
        self._padding = padding

    def _retrieve_elements_from_indices(self, tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output

    def forward(self, inputs_mean, inputs_variance):
        inputs_mean = F.pad(inputs_mean, (self._padding, self._padding))
        inputs_variance = F.pad(inputs_variance, (self._padding, self._padding))
        outputs_mean, indices = self.maxpool(inputs_mean)
        outputs_variance = self._retrieve_elements_from_indices(inputs_variance, indices)
        return outputs_mean, outputs_variance


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2d, self).__init__()

    def forward(self, inputs_mean, inputs_variance):
        batch_size = inputs_mean.size(0)
        channels = inputs_mean.size(1)
        outputs_mean = inputs_mean.contiguous().view(batch_size, channels, -1)
        outputs_variance = inputs_variance.contiguous().view(batch_size, channels, -1) 
        N = outputs_mean.size(2)
        outputs_mean = outputs_mean.mean(dim=2)
        outputs_variance = outputs_variance.mean(dim=2) * (1.0 / float(N))
        return outputs_mean, outputs_variance

    def forward_sampling(self, inputs):
        batch_size = inputs.size(0)
        channels = inputs.size(1)
        outputs = inputs.contiguous().view(batch_size, channels, -1)
        outputs = outputs.mean(dim=2)
        return outputs


class Sequential(nn.Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, inputs, inputs_variance):
        for module in self._modules.values():
            inputs, inputs_variance = module(inputs, inputs_variance)

        return inputs, inputs_variance

    def forward_sampling(self, inputs):
        for module in self._modules.values():
            inputs = module.forward_sampling(inputs)
        return inputs

    