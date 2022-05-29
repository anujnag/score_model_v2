import jax
import jax.numpy as jnp
import haiku as hk


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    return jnp.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / jnp.log(sigma))


class NCSN(hk.Module):
    def __init__(self, cfg):
        super().__init__(name=None)
        self.layer_sizes = cfg.layer_sizes
        self.embed_dim = cfg.embed_dim

    def __call__(self, x, t, sigma):

        
        act = jax.nn.swish
        W = hk.get_parameter(
            "W", (1, self.embed_dim // 2), init=hk.initializers.RandomNormal(sigma)
        )
        W = jax.lax.stop_gradient(W)
        # t_proj = t * W * 2 * jnp.pi
        # t_proj = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        # std_embedding3 = hk.Linear(self.embed_dim)(t_proj)

        t_proj = t[:, None, None, None] * W[None, None, None, :] * 2 * jnp.pi
        t_proj = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        std_embedding = hk.Linear(self.embed_dim)(t_proj)
        std_embedding3 = act(std_embedding)



        x = jnp.concatenate([x, std_embedding3], axis=-1)
        x = hk.nets.MLP(self.layer_sizes)(x)
        return x
