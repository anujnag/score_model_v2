import flax.linen as nn
import functools
import haiku as hk
import jax.nn.initializers as init
import jax
import jax.numpy as jnp
import pdb
from layers import (RefineBlock, ResidualBlock, ncsn_conv3x3)

conv3x3 = ncsn_conv3x3

class NCSNv2(hk.Module):
  """NCSNv2 model architecture."""
  def __init__(self, cfg):
    super().__init__(name=None)
    if cfg.dataset == 'mnist':
      self.image_channels = 1
    elif cfg.dataset == 'cifar10':
      self.image_channels = 3
    else:
      self.image_channels = 1    
 
  @nn.compact
  def __call__(self, x, t, sigma, train=True):
    # config parsing
    # config = self.config
    nf = 128
    act = jax.nn.swish
    normalizer = hk.GroupNorm
    sigmas = jnp.exp(jnp.linspace(jnp.log(50), jnp.log(0.01), 1000))
    interpolation = 'bilinear'

    h = conv3x3(x, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer, dilation=1)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer, dilation=1)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer, dilation=1)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=1)(h)
    
    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=2)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)

    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=4)(layer3)
    layer4 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    
    # U-Net with RefineBlocks
    ref1 = RefineBlock(layer4.shape[1:3],
                       2 * nf,
                       act=act,
                       interpolation=interpolation,
                       start=True,
                       end=False)([layer4])
    ref2 = RefineBlock(layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=False,
                       end=False)([layer3, ref1])
    ref3 = RefineBlock(layer2.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=False,
                       end=False)([layer2, ref2])
    ref4 = RefineBlock(layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       start=False,
                       end=True)([layer1, ref3])

    h = normalizer(32)(ref4)
    h = act(h)
    h = conv3x3(h, self.image_channels)

    # used_sigmas = sigmas[labels].reshape(
    #     (x.shape[0], *([1] * len(x.shape[1:]))))
    
    return h # / used_sigmas