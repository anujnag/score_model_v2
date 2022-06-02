import functools
from typing import Any, Sequence, Optional
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk

def ncsn_conv3x3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
    """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
    init_scale = 1e-10 if init_scale == 0 else init_scale
    kernel_init = hk.initializers.VarianceScaling(
        scale=1 / 3 * init_scale, mode='fan_in', distribution='uniform')
    kernel_shape = (3, 3) + (x.shape[-1], out_planes)
    bias_init = hk.initializers.VarianceScaling(
        scale=1 / 3 * init_scale, mode='fan_in', distribution='uniform')
    output = hk.Conv2D(
        out_planes,
        (3, 3),
        stride=(stride, stride),
        rate=(dilation, dilation),
        padding='SAME',
        with_bias=bias,
        w_init=kernel_init,
        b_init=bias_init
    )(x)
    
    return output

class RCUBlock(hk.Module):
    """RCUBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, features, n_blocks, n_stages, act):
        super().__init__(name=None)
        self.features = features
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def __call__(self, x):
        for _ in range(self.n_blocks):
            residual = x
            for _ in range(self.n_stages):
                x = self.act(x)
                x = ncsn_conv3x3(x, self.features, stride=1, bias=False)
            x = x + residual

        return x

class ConvMeanPool(hk.Module):
    """ConvMeanPool for building the ResNet backbone."""
    def __init__(self, output_dim, kernel_size, biases):
        super().__init__(name=None)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.biases = biases

    def __call__(self, inputs):
        output = hk.Conv2D(
            self.output_dim,
            (self.kernel_size, self.kernel_size),
            stride=(1, 1),
            padding='SAME',
            with_bias=self.biases
        )(inputs)

        output = sum([
        output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
        output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]
        ]) / 4.
        return output

class CRPBlock(hk.Module):
    """CRPBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, features, n_stages, act):
        super().__init__(name=None)
        self.features = features
        self.n_stages = n_stages
        self.act = act

    def __call__(self, x):
        x = self.act(x)
        path = x
        for _ in range(self.n_stages):
            path = nn.max_pool(
                path, window_shape=(5, 5), strides=(1, 1), padding='SAME')
            path = ncsn_conv3x3(path, self.features, stride=1, bias=False)
            x = path + x
        return x

class MSFBlock(hk.Module):
    """MSFBlock for RefineNet. Used in NCSNv2."""
    def __init__(self, shape, features, interpolation):
        super().__init__(name=None)
        self.shape = shape
        self.features = features
        self.interpolation = interpolation

    def __call__(self, xs):
        sums = jnp.zeros((xs[0].shape[0], *self.shape, self.features))
        for i in range(len(xs)):
            h = ncsn_conv3x3(xs[i], self.features, stride=1, bias=True)
            if self.interpolation == 'bilinear':
                h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'bilinear')
            elif self.interpolation == 'nearest_neighbor':
                h = jax.image.resize(h, (h.shape[0], *self.shape, h.shape[-1]), 'nearest')
            else:
                raise ValueError(f'Interpolation {self.interpolation} does not exist!')
            sums = sums + h
        return sums

class ResidualBlock(hk.Module):
    """The residual block for defining the ResNet backbone. Used in NCSNv2."""
    def __init__(self, output_dim, normalization, resample, act, dilation):
        super().__init__(name=None)
        self.output_dim = output_dim
        self.normalization = normalization
        self.resample = resample
        self.act = act
        self.dilation = dilation

    def __call__(self, x):
        h = self.normalization(32)(x)
        h = self.act(h)
        if self.resample == 'down':
            h = ncsn_conv3x3(h, h.shape[-1], dilation=self.dilation)
            h = self.normalization(32)(h)
            h = self.act(h)
            if self.dilation > 1:
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
            else:
                h = ConvMeanPool(output_dim=self.output_dim, kernel_size=3, biases=True)(h)
                shortcut = ConvMeanPool(output_dim=self.output_dim, kernel_size=1, biases=True)(x)
        elif self.resample is None:
            if self.dilation > 1:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv3x3(x, self.output_dim, dilation=self.dilation)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
                h = self.normalization(32)(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim, dilation=self.dilation)
            else:
                if self.output_dim == x.shape[-1]:
                    shortcut = x
                else:
                    shortcut = ncsn_conv1x1(x, self.output_dim)
                h = ncsn_conv3x3(h, self.output_dim)
                h = self.normalization(32)(h)
                h = self.act(h)
                h = ncsn_conv3x3(h, self.output_dim)

        return h + shortcut

class RefineBlock(hk.Module):
    """RefineBlock for building NCSNv2 RefineNet."""
    def __init__(self, output_shape, features, act, interpolation, start, end):
        super().__init__(name=None)
        self.output_shape = output_shape
        self.features = features
        self.act = act
        self.interpolation = interpolation
        self.start = start
        self.end = end

    def __call__(self, xs):
        rcu_block = functools.partial(RCUBlock, n_blocks=2, n_stages=2, act=self.act)
        rcu_block_output = functools.partial(RCUBlock,
                                            features=self.features,
                                            n_blocks=3 if self.end else 1,
                                            n_stages=2,
                                            act=self.act)
        hs = []
        for i in range(len(xs)):
            h = rcu_block(features=xs[i].shape[-1])(xs[i])
            hs.append(h)

        if not self.start:
            msf = functools.partial(MSFBlock, features=self.features, interpolation=self.interpolation)
            h = msf(shape=self.output_shape)(hs)
        else:
            h = hs[0]

        crp = functools.partial(CRPBlock, features=self.features, n_stages=2, act=self.act)
        h = crp()(h)
        h = rcu_block_output()(h)
        return h
