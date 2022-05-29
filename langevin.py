import jax
import jax.numpy as jnp
from functools import partial
from jax import random, vmap, jit
from NCSN import marginal_prob_std
import pdb

global_sigma = 25
epsilon, n_iterations, n_burn_in = 1e-4, 5000, 4000


def itemwise_f(f, params, state, x, t, sigma):
    """Calls f with a single x, t, handling reshaping
    which is necesary for the batchnorm"""
    assert len(x.shape) == 3
    assert len(t.shape) == 1 or len(t.shape) == 0
    expanded_x = jnp.expand_dims(x, axis=0)
    expanded_t = jnp.expand_dims(t, axis=0)
    return f.apply(params, state, expanded_x, expanded_t, sigma, is_training=False)


def langevin_wrapper(f, params, state, key, x_0, t_array):
    @jit
    def annealed_langevin(params, state, key, x_0, t_array):
        def langevin(carry, t_array):
            x_0, state, key = carry
            t = t_array
            key, _ = random.split(key)

            def inner_fn(carry, key):
                x, state = carry
                key, _ = random.split(key)
                out, state = itemwise_f(f, params, state, x, t, global_sigma)
                score = out.squeeze(axis=0)
                x = (
                    x
                    + epsilon * score
                    + jnp.sqrt(2 * epsilon) * random.normal(key, shape=x.shape)
                )
                return (x, state), x

            (x, state), xs = jax.lax.scan(
                inner_fn, (x_0, state), random.split(key, n_iterations)
            )
            return (x, state, key), xs

        (x, state, key), xs = jax.lax.scan(langevin, (x_0, state, key), t_array)
        return xs[-1, n_burn_in:, :]

    chain_langevin = vmap(annealed_langevin, (None, None, 0, 0, None))
    ret = chain_langevin(params, state, key, x_0, t_array)
    return ret
