import jax
from jax._src.api import jit
from jax.experimental.ode import odeint
import jax.numpy as jnp
import numpy as np
import torch
from config import get_config
import pdb
from UNet import marginal_prob_std
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import tqdm
import wandb

conf = get_config()


def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
    standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[2:])
    return -N / 2.0 * jnp.log(2 * np.pi * sigma ** 2) - jnp.sum(
        z ** 2, axis=(2, 3, 4)
    ) / (2 * sigma ** 2)


def score_fn(cfg, f, params, state, x, t):
    perturbed_x = jnp.concatenate([x] * cfg.num_samples, axis=-1)
    t = jnp.squeeze(t, axis=1)
    out = f.apply(params, state, perturbed_x, t, cfg.sigma, True)
    # KEEP THIS FOR NCSN
    score = out[0]
    # -----------------
    # KEEP THIS FOR DENOISER
    # score = (jnp.concatenate([out[0]] * cfg.num_samples, axis=-1) - perturbed_x) / (
    #     marginal_prob_std(t, cfg.sigma) ** 2
    # )[:, None, None, None]
    # score = score[:, :, :, :1]
    # score = jnp.mean(score, axis=-1)[:, :, :, None]
    # -------------------------------
    return score


def ode_likelihood(cfg, f, rng, x, params, state, batch_size, eps=1e-5, num_samples=1):
    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    rng, step_rng = jax.random.split(rng)

    def divergence_eval(sample, time_steps):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        epsilon = jax.random.normal(rng, sample.shape)
        score_e_fn = lambda x: jnp.sum(
            score_fn(cfg, f, params, state, x, time_steps) * epsilon
        )
        grad_score_e = jax.grad(score_e_fn)(sample)
        return jnp.sum(grad_score_e * epsilon, axis=(1, 2, 3))

    shape = x.shape
    time_shape = (shape[0], shape[1])

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        sample = jnp.squeeze(sample, axis=1)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        score = score_fn(cfg, f, params, state, sample, time_steps)
        return jnp.asarray(score).reshape((-1,)).astype(np.float32)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        # Obtain x(t) by solving the probability flow ODE.
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        sample = jnp.squeeze(sample, axis=1)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        # Compute likelihood.
        div = divergence_eval(sample, time_steps)
        return jnp.asarray(div).reshape((-1,)).astype(np.float32)

    @jit
    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones(time_shape) * t
        sample = x[: -shape[0] * shape[1]]
        logp = x[-shape[0] * shape[1] :]
        g = cfg.sigma ** t
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return jnp.concatenate([sample_grad, logp_grad], axis=0)

    # Run the black-box ODE solver.
    # jax_odeint_fn = lambda y, t: ode_func(t, y)

    @jit
    def jax_odeint_fn(y, t):
        return ode_func(t, y)

    init_x = jnp.concatenate(
        [x.reshape((-1,)), jnp.zeros((shape[0] * shape[1],))], axis=0
    )

    res_jax = odeint(
        jax_odeint_fn,
        jnp.array(init_x),
        jnp.linspace(eps, 1.0),
        atol=1e-5,
    )
    # zp = jnp.asarray(res.y[:, -1])
    zp = res_jax[-1]
    z = zp[: -shape[0] * shape[1]].reshape(shape)
    delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
    sigma_max = marginal_prob_std(1.0, cfg.sigma)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[2:])
    bpd = bpd / N + 8.0
    return z, bpd


def likelihood_wrapper(f, cfg, params, state):
    sample_batch_size = 64
    dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)
    dataset = torch.utils.data.Subset(dataset, torch.arange(500))
    data_loader = DataLoader(
        dataset,
        batch_size=sample_batch_size,
        shuffle=True,
    )

    all_bpds = 0.0
    all_items = 0
    rng = jax.random.PRNGKey(100)
    tqdm_data = tqdm.tqdm(data_loader)
    for x, _ in tqdm_data:
        x = x.permute(0, 2, 3, 1).cpu().numpy().reshape((-1, 1, 28, 28, 1))
        # x = x.permute(0, 2, 3, 1).cpu().numpy().reshape((-1, 1, 32, 32, 3))
        rng, step_rng = jax.random.split(rng)
        z = jax.random.uniform(step_rng, x.shape)
        x = (x * 255.0 + z) / 256.0
        _, bpd = ode_likelihood(
            cfg, f, step_rng, x, params, state, sample_batch_size, eps=1e-5
        )
        all_bpds += bpd.sum()
        all_items += bpd.shape[0] * bpd.shape[1]
        tqdm_data.set_description(
            "Average bits/dim: {:5f}".format(all_bpds / all_items)
        )
        if cfg.use_wandb:
            wandb.log(
                data={
                    "Average bits/dim: ": float(all_bpds / all_items),
                },
            )

    return all_bpds / all_items
