import jax.numpy as jnp
from langevin import langevin_wrapper
import matplotlib.pyplot as plt
import jax.random as rnd
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pickle
from torchvision.utils import make_grid
from torchvision.utils import save_image
import hydra
from omegaconf import DictConfig, OmegaConf
from train import train_wrapper
from models import get_model
import pdb
from likelihood import likelihood_wrapper
import wandb
from UNet import marginal_prob_std
import torch
import numpy as np
from ode_sampler import ode_sampler


@hydra.main(config_path=".", config_name="debug_config")
def main(cfg: DictConfig) -> None:

    train_dataset = MNIST(
        "../../..", train=True, transform=transforms.ToTensor(), download=True
    )
    # train_dataset = CIFAR10(
    #     "../../..", train=True, transform=transforms.ToTensor(), download=True
    # )
    # train_dataset = Subset(train_dataset, torch.arange(100))
    # idx = train_dataset.targets > 1
    # train_dataset.data, train_dataset.targets = (
    #     train_dataset.data[idx],
    #     train_dataset.targets[idx],
    # )
    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    test_dataset = MNIST(
        "../../..", train=False, transform=transforms.ToTensor(), download=False
    )
    # test_dataset = CIFAR10(
    #     "../../..", train=False, transform=transforms.ToTensor(), download=False
    # )
    test_data_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    if cfg.use_wandb:
        wandb.init(project="ncsnpp", config=cfg)

    if cfg.load_model:
        with open(
            f"../../../model_weights/dae_model_{cfg.num_samples}samples.pickle", "rb"
        ) as handle:
            params, state = pickle.load(handle)
    else:
        params, state = train_wrapper(train_data_loader, test_data_loader, cfg)
        print("Model training done\n")

        with open(
            f"../../../model_weights/dae_mnist_consistency.pickle", "wb"
        ) as handle:
            pickle.dump((params, state), handle)

    _, _, f = get_model(cfg)

    # Langevin Chain
    if cfg.langevin:
        init_x = (
            rnd.normal(
                rnd.PRNGKey(0),
                (cfg.chain_length, 28, 28, cfg.num_samples),
            )
            * marginal_prob_std(1.0, cfg.sigma)
        )
        key_array = rnd.split(rnd.PRNGKey(0), init_x.shape[0])
        out = langevin_wrapper(
            f,
            params,
            state,
            key_array,
            init_x,
            jnp.flip(jnp.array([1e-3, 5e-3, 25e-3, 125e-3, 625e-3, 1])),
        )
        out = jnp.clip(out, 0.0, 1.0)
        grid_img = np.array(out)
        grid_img = torch.from_numpy(grid_img)
        indices = np.arange(0, grid_img.shape[1], grid_img.shape[1] / 5)
        grid_img = grid_img[:, indices, :, :, :]
        grid_img = torch.reshape(grid_img, (-1, 28, 28, 1))
        # grid_img = torch.reshape(grid_img, (-1, 32, 32, 3))
        grid_img = torch.permute(grid_img, (0, 3, 1, 2))
        save_image(grid_img, fp=f"../../../images/MNIST_langevin.png", nrow=5)
        if cfg.use_wandb:
            wandb.log(
                {"training_figure": wandb.Image(f"../../../images/MNIST_langevin.png")}
            )

    if cfg.plot_points:
        num_images = 5
        _, key = rnd.split(rnd.PRNGKey(0))
        init_x = rnd.normal(key, (num_images, 28, 28, 1)) * marginal_prob_std(
            1.0, cfg.sigma
        )
        out = ode_sampler(
            f,
            params,
            state,
            init_x,
            cfg.sigma,
            error_tol=1e-2,
            chain_length=num_images,
            num_samples=cfg.num_samples,
        )
        out = jnp.clip(out, 0.0, 1.0)
        grid_img = np.array(out)
        grid_img = torch.from_numpy(grid_img)
        grid_img = torch.permute(grid_img, (0, 3, 1, 2))
        grid_img = grid_img[:, :1, :, :]
        # save_image(
        #     grid_img,
        #     fp=f"../../../images/MNIST_{cfg.num_samples}.png",
        #     nrow=grid_img.shape[0],
        # )
        save_image(
            grid_img,
            fp=f"../../../images/MNIST_consistency.png",
            nrow=grid_img.shape[0],
        )
        if cfg.use_wandb:
            wandb.log(
                {
                    "training_figure": wandb.Image(
                        f"../../../images/MNIST_consistency.png"
                    )
                }
            )
    if cfg.calculate_likelihood:
        bpd = likelihood_wrapper(f, cfg, params, state)
        print(f"Final bpd: {bpd}")


if __name__ == "__main__":
    main()
