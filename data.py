import tensorflow_probability.substrates.jax.distributions as tfd
from config import get_config
from jax import random, jit
from typing import List
import jax.numpy as jnp

conf = get_config()


def get_gaussian_mixture(
    means: List[jnp.ndarray], std: float, weights: List[float]
) -> tfd.MixtureSameFamily:
    """Get mixture of isotropic gaussian distributions
    Args:
        means: List of mean locations
        std: Common standard deviation for each component
        weights: Respective weight in mixture distribution
    Returns: tfp distribution object giving the mixture of Gaussians
    """
    data_dim = len(means[0])
    assert len(means) == len(weights)
    assert all([x.shape == (data_dim,) for x in means])
    assert jnp.allclose(jnp.sum(jnp.array(weights)), 1)

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights),
        components_distribution=tfd.MultivariateNormalDiag(
            loc=means, scale_identity_multiplier=[std] * data_dim,
        ),
    )
    return gm


# def make_data():
#     gm = tfd.MixtureSameFamily(
#         mixture_distribution=tfd.Categorical(probs=[0.8, 0.2]),
#         components_distribution=tfd.MultivariateNormalDiag(
#             loc=[[-4.0] * conf.data_dim, [4.0] * conf.data_dim],
#             scale_identity_multiplier=[1.0] * conf.data_dim,
#         ),
#     )

#     x = jit(gm.sample, static_argnames=("sample_shape"))(
#         seed=random.PRNGKey(0), sample_shape=(conf.n_data,)
#     )

#     return x
