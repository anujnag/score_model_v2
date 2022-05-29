from jax import random
from typing import NamedTuple, List
from jax import numpy as jnp
from jax._src.prng import PRNGKeyArray


# Changed this from ml_collections since ml_collections
# gave a load of errors on my type checker (pyright)


class Config(NamedTuple):
    data_dim: int
    num_samples: int
    n_data: int
    layer_sizes: List[int]
    channels: List[int]
    scale: int
    embed_dim: int
    key: PRNGKeyArray
    batch_size: int
    chain_length: int
    sigma: int
    num_epochs: int
    lr: float
    error_tolerance: float
    langevin_iterations: int
    langevin_burnin: int
    langevin_stepsize: float
    use_wandb: bool
    load_model: bool


def get_config():
    num_samples: int = 1
    data_dim: int = 2

    return Config(
        data_dim=data_dim,
        num_samples=num_samples,
        n_data=3_000,
        layer_sizes=[32, 32, 32, data_dim],
        channels=[32, 64, 128, 256],
        scale=30,
        embed_dim=256,
        key=random.PRNGKey(1),
        batch_size=256,
        chain_length=1,
        sigma=25,
        num_epochs=40,
        lr=1e-4,
        error_tolerance=1e-5,
        langevin_iterations=100_000,
        langevin_burnin=80_000,
        langevin_stepsize=1e-4,
        use_wandb=False,
        load_model=False,
    )
