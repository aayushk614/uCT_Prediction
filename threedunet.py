
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

import collections
import contextlib

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None

try:
    from jactorch.parallel.comm import SyncMaster
    from jactorch.parallel.data_parallel import JacDataParallel as DataParallelWithCallback
except ImportError:
    from comm import SyncMaster
    from replicate import DataParallelWithCallback

__all__ = [
    'SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d',
    'patch_sync_batchnorm', 'convert_model'
]


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'

        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape::
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


@contextlib.contextmanager
def patch_sync_batchnorm():
    import torch.nn as nn

    backup = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d

    nn.BatchNorm1d = SynchronizedBatchNorm1d
    nn.BatchNorm2d = SynchronizedBatchNorm2d
    nn.BatchNorm3d = SynchronizedBatchNorm3d

    yield

    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d = backup


def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d
       to SynchronizedBatchNorm*N*d

    Args:
        module: the input module needs to be convert to SyncBN model

    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using SyncBN
        >>> m = convert_model(m)
    """
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = DataParallelWithCallback(mod)
        return mod

    mod = module
    for pth_module, sync_module in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                        torch.nn.modules.batchnorm.BatchNorm2d,
                                        torch.nn.modules.batchnorm.BatchNorm3d],
                                       [SynchronizedBatchNorm1d,
                                        SynchronizedBatchNorm2d,
                                        SynchronizedBatchNorm3d]):
        if isinstance(module, pth_module):
            mod = sync_module(module.num_features, module.eps, module.momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod


# In[3]:


from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch


class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)
        
        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm3d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


# In[4]:


import os,sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F



# common layers
def get_functional_act(mode='relu'):
    if mode == 'relu':
        return F.relu(inplace=True)
    elif mode == 'elu':
        return F.elu(inplace=True)
    elif mode[:5] == 'leaky':
        # 'leaky0.2' 
        return F.leaky_relu(inplace=True, negative_slope=float(mode[5:]))
    raise ValueError('Unknown activation functional option {}'.format(mode))

def get_layer_act(mode=''):
    if mode == '':
        return []
    elif mode == 'relu':
        return [nn.ReLU(inplace=True)]
    elif mode == 'elu':
        return [nn.ELU(inplace=True)]
    elif mode[:5] == 'leaky':
        # 'leaky0.2' 
        return [nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))]
    raise ValueError('Unknown activation layer option {}'.format(mode))

def get_layer_norm(out_planes, norm_mode='', dim=2):
    if norm_mode=='':
        return []
    elif norm_mode=='bn':
        if dim==1:
            return [SynchronizedBatchNorm1d(out_planes)]
        elif dim==2:
            return [SynchronizedBatchNorm2d(out_planes)]
        elif dim==3:
            return [SynchronizedBatchNorm3d(out_planes)]
    elif norm_mode=='abn':
        if dim==1:
            return [nn.BatchNorm1d(out_planes)]
        elif dim==2:
            return [nn.BatchNorm2d(out_planes)]
        elif dim==3:
            return [nn.BatchNorm3d(out_planes)]
    elif norm_mode=='in':
        if dim==1:
            return [nn.InstanceNorm1d(out_planes)]
        elif dim==2:
            return [nn.InstanceNorm2d(out_planes)]
        elif dim==3:
            return [nn.InstanceNorm3d(out_planes)]
    elif norm_mode=='bin':
        if dim==1:
            return [BatchInstanceNorm1d(out_planes)]
        elif dim==2:
            return [BatchInstanceNorm2d(out_planes)]
        elif dim==3:
            return [BatchInstanceNorm3d(out_planes)]
    raise ValueError('Unknown normalization norm option {}'.format(mode))


# conv basic blocks
def conv2d_norm_act(in_planes, out_planes, kernel_size=(3,3), stride=1, 
                  dilation=(1,1), padding=(1,1), bias=True, pad_mode='rep', norm_mode='', act_mode='', return_list=False):

    if isinstance(padding,int):
        pad_mode = pad_mode if padding!=0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding)!=0 else 'zeros'

    if pad_mode in ['zeros','circular']:
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)] 
    elif pad_mode=='rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad2d(padding),
                  nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))
    layers += get_layer_norm(out_planes, norm_mode)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)

def conv3d_norm_act(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=True, pad_mode='rep', norm_mode='', act_mode='', return_list=False):
    
    if isinstance(padding,int):
        pad_mode = pad_mode if padding!=0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding)!=0 else 'zeros'

    if pad_mode in ['zeros','circular']:
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)] 
    elif pad_mode=='rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad3d(padding),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))

    layers += get_layer_norm(out_planes, norm_mode, 3)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)


# In[5]:


# 1. Residual blocks
# implemented with 2D conv
class residual_block_2d_c2(nn.Module):
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d_c2, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act( in_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3,3), padding=(1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1,1), padding=(0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y  

# implemented with 3D conv
class residual_block_2d(nn.Module):
    """
    Residual Block 2D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=True, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,3,3), padding=(0,1,1), pad_mode=pad_mode,norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y  

class residual_block_3d(nn.Module):
    """Residual Block 3D

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
    """
    def __init__(self, in_planes, out_planes, projection=False, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(residual_block_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), padding=(1,1,1), pad_mode=pad_mode, norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]
        
    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y       

class bottleneck_dilated_2d(nn.Module):
    """Bottleneck Residual Block 2D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_2d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(3, 3), dilation=(dilate, dilate), padding=(dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv2d_norm_act(out_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        )
        self.projector = conv2d_norm_act(in_planes, out_planes, kernel_size=(1, 1), padding=(0, 0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.elu(y)
        return y

class bottleneck_dilated_3d(nn.Module):
    """Bottleneck Residual Block 3D with Dilated Convolution

    Args:
        in_planes (int): number of input channels.
        out_planes (int): number of output channels.
        projection (bool): projection of the input with a conv layer.
        dilate (int): dilation rate of conv filters.
    """
    def __init__(self, in_planes, out_planes, projection=False, dilate=2, pad_mode='rep', norm_mode='bn', act_mode='elu'):
        super(bottleneck_dilated_3d, self).__init__()
        self.projection = projection
        self.conv = nn.Sequential(
            conv3d_norm_act( in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(3,3,3), 
                          dilation=(1, dilate, dilate), padding=(1, dilate, dilate), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            conv3d_norm_act(out_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        )
        self.projector = conv3d_norm_act(in_planes, out_planes, kernel_size=(1,1,1), padding=(0,0,0), norm_mode=norm_mode)
        self.act = get_layer_act(act_mode)[0]

    def forward(self, x):
        y = self.conv(x)
        if self.projection:
            y = y + self.projector(x)
        else:
            y = y + x
        y = self.act(y)
        return y        


# In[6]:


def conv3d_norm_act(in_planes, out_planes, kernel_size=(3,3,3), stride=1, 
                  dilation=(1,1,1), padding=(1,1,1), bias=True, pad_mode='rep', norm_mode='', act_mode='', return_list=False):
    
    if isinstance(padding,int):
        pad_mode = pad_mode if padding!=0 else 'zeros'
    else:
        pad_mode = pad_mode if max(padding)!=0 else 'zeros'

    if pad_mode in ['zeros','circular']:
        layers = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=padding, padding_mode=pad_mode, dilation=dilation, bias=bias)] 
    elif pad_mode=='rep':
        # the size of the padding should be a 6-tuple        
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        layers = [nn.ReplicationPad3d(padding),
                  nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                     stride=stride, padding=0, dilation=dilation, bias=bias)]
    else:
        raise ValueError('Unknown padding option {}'.format(mode))

    layers += get_layer_norm(out_planes, norm_mode, 3)
    layers += get_layer_act(act_mode)
    if return_list:
        return layers
    else:
        return nn.Sequential(*layers)

    


# In[7]:


import torch
import torch.nn as nn
from math import sqrt

def xavier_init(model):
    # default xavier initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform(
                m.weight(), gain=nn.init.calculate_gain('relu'))

def he_init(model):
    # he initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal(m.weight, mode='fan_in')

def selu_init(model):
    # selu init
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))
        elif isinstance(m, nn.Linear):
            fan_in = m.in_features
            nn.init.normal(m.weight, 0, sqrt(1. / fan_in))

def ortho_init(model):
    # orthogonal initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)


# In[8]:


class unet_residual_3d(nn.Module):
    """Lightweight 3D U-net with residual blocks (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.
        
    Args:
        in_channel (int): number of input channels.
        out_channel (int): number of output channels.
        filters (list): number of filters at each u-net stage.
    """
    def __init__(self, in_channel=1, out_channel=13, filters=[28, 36, 48, 64, 80], pad_mode='rep', norm_mode='bn', act_mode='elu', do_embedding=True, head_depth=1):
        super().__init__()

        self.depth = len(filters)-2
        self.do_embedding = do_embedding

        # encoding path
        if self.do_embedding: 
            num_out = filters[1]
            self.downE = nn.Sequential(
                # anisotropic embedding
                conv3d_norm_act(in_planes=in_channel, out_planes=filters[0], 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                # 2d residual module
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_2d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            )
        else:
            filters[0] = in_channel
            num_out = out_channel
        
        self.downC = nn.ModuleList(
            [nn.Sequential(
            conv3d_norm_act(in_planes=filters[x], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[x+1], filters[x+1], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode)
            ) for x in range(self.depth)])

        # pooling downsample
        self.downS = nn.ModuleList([nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) for x in range(self.depth+1)])

        # center block
        self.center = nn.Sequential(conv3d_norm_act(in_planes=filters[-2], out_planes=filters[-1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
            residual_block_3d(filters[-1], filters[-1], projection=True)
        )

        self.upC = nn.ModuleList(
            [nn.Sequential(
                conv3d_norm_act(in_planes=filters[x+1], out_planes=filters[x+1], 
                          kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_3d(filters[x+1], filters[x+1], projection=False)
            ) for x in range(self.depth)])

        if self.do_embedding: 
            # decoding path
            self.upE = nn.Sequential(
                conv3d_norm_act(in_planes=filters[0], out_planes=filters[0], 
                              kernel_size=(1,3,3), stride=1, padding=(0,1,1), pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                residual_block_2d(filters[0], filters[0], projection=False, pad_mode=pad_mode, norm_mode=norm_mode, act_mode=act_mode),
                conv3d_norm_act(in_planes=filters[0], out_planes=out_channel, 
                              kernel_size=(1,5,5), stride=1, padding=(0,2,2), pad_mode=pad_mode, norm_mode=norm_mode)
            )
            # conv + upsample
            self.upS = nn.ModuleList([nn.Sequential(
                            conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(self.depth+1)])
        else:
            # new
            head_pred = [residual_block_3d(filters[1], filters[1], projection=False)
                                    for x in range(head_depth-1)] + \
                              [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)]
            self.upS = nn.ModuleList( [nn.Sequential(*head_pred)] +                                  [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                        nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            """
            # old
            self.upS = nn.ModuleList( [conv3d_norm_act(filters[1], out_channel, kernel_size=(1,1,1), padding=0, norm_mode=norm_mode)] + \
                                 [nn.Sequential(
                        conv3d_norm_act(filters[x+1], filters[x], kernel_size=(1,1,1), padding=0, norm_mode=norm_mode, act_mode=act_mode),
                        nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)) for x in range(1,self.depth+1)])
            """


        #initialization
        ortho_init(self)

    def forward(self, x):
        # encoding path
        if self.do_embedding: 
            z = self.downE(x)
            x = self.downS[0](z)

        down_u = [None] * (self.depth)
        for i in range(self.depth):
            down_u[i] = self.downC[i](x)
            x = self.downS[i+1](down_u[i])

        x = self.center(x)

        # decoding path
        for i in range(self.depth-1,-1,-1):
            x = down_u[i] + self.upS[i+1](x)
            x = self.upC[i](x)

        if self.do_embedding: 
            x = z + self.upS[0](x)
            x = self.upE(x)
        else:
            x = self.upS[0](x)

        x = torch.sigmoid(x)
        return x






