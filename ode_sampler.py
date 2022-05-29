from scipy import integrate
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from jax import jit
import pdb
from UNet import marginal_prob_std


def ode_sampler(
    f,
    params,
    state,
    init_x,
    global_sigma,
    eps=1e-5,
    chain_length=1,
    error_tol=1e-4,
    num_samples=1,
):
    time_shape = (chain_length,)
    # sample_shape = (chain_length, 32, 32, num_samples)
    sample_shape = (chain_length, 28, 28, 1)

    def jax_score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = sample.reshape(sample_shape)
        time_steps = time_steps.reshape(time_shape)
        out = f.apply(params, state, sample, time_steps, global_sigma, False)
        score = out[0]
        # score = (
        #     jnp.concatenate([out[0]] * num_samples, axis=-1) - sample
        # ) / marginal_prob_std(time_steps, global_sigma) ** 2
        return jnp.asarray(score).reshape((-1,)).astype(jnp.float32)

    @jit
    def jax_ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        t = 1.0 - t
        time_steps = np.ones(time_shape) * t
        g = global_sigma ** t
        # return -0.5 * (g ** 2) * jax_score_eval_wrapper(x, time_steps)
        # below line is for reverse process (sampling) and
        # above line for forward process
        return 0.5 * (g ** 2) * jax_score_eval_wrapper(x, time_steps)

    jax_odeint_fn = lambda y, t: jax_ode_func(t, y)
    result_0 = odeint(
        jax_odeint_fn,
        jnp.array(init_x).reshape(-1),
        jnp.linspace(0, 1 - eps),
        atol=error_tol,
    )
    return result_0[-1].reshape(sample_shape)


# batch_ode = lambda
