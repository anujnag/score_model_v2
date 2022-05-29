from ode_sampler import ode_sampler
import pickle
import torch
from torchvision.transforms import RandAugment
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import pdb
import matplotlib.pyplot as plt
import tqdm
import warnings
import wandb

warnings.filterwarnings("ignore")

torch.manual_seed(8)


def augmentation(x):
    x = torch.tensor(255 * x, dtype=torch.uint8)
    x = torch.permute(x, (0, 3, 1, 2))
    aux = RandAugment(3, 5, 31).forward(x)
    aug_x = torch.tensor(aux / 255.0, dtype=torch.float32)
    aug_x = torch.permute(aug_x, (0, 2, 3, 1))
    return aug_x


@hydra.main(config_path=".", config_name="debug_config")
def main(cfg: DictConfig) -> None:
    if cfg.use_wandb:
        wandb.init(project="ncsnpp", config=cfg)
    data_shape = (-1, 28, 28, 1)

    train_dataset = MNIST(
        "../../..", train=True, transform=transforms.ToTensor(), download=True
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    _, _, f = get_model(cfg)

    with open(
        f"../../../model_weights/dae_model_{cfg.num_samples}samples.pickle", "rb"
    ) as handle:
        params, state = pickle.load(handle)

    # tqdm_data = tqdm.tqdm(train_data_loader)
    # num_comparisons, all_distorg, all_distaug = 0, 0, 0
    # for x, _ in tqdm_data:
    #     # for j, (x, _) in enumerate(train_data_loader2):
    #     x0 = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
    #     aug0 = augmentation(x0).numpy()
    #     augT = ode_sampler(
    #         f, params, state, aug0, 25.0, 1e-3, chain_length=cfg.batch_size
    #     )
    #     xT = ode_sampler(f, params, state, x0, 25.0, 1e-3, chain_length=cfg.batch_size)
    #     distorg = np.mean(np.linalg.norm(x0 - aug0, axis=(1, 2)))
    #     distaug = np.mean(np.linalg.norm(xT - augT, axis=(1, 2)))
    #     all_distorg += distorg
    #     all_distaug += distaug
    #     num_comparisons += 1

    #     wandb.log(
    #         data={
    #             "Mean distance between x(0) and aug(0)": float(
    #                 all_distorg / num_comparisons
    #             )
    #         }
    #     )
    #     wandb.log(
    #         data={
    #             "Mean distance between x(T) and aug(T)": float(
    #                 all_distaug / num_comparisons
    #             )
    #         }
    #     )
    #     tqdm_data.set_description(
    #         "Mean distance between x(0) and y(0): {:5f}".format(
    #             all_distorg / num_comparisons
    #         )
    #     )

    #     tqdm_data.set_description(
    #         "Mean distance between x(T) and aug(T): {:5f}".format(
    #             all_distaug / num_comparisons
    #         )
    #     )

    train_data_loader2 = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2
    )
    num_comparisons, all_distorg, all_distaug = 0, 0, 0
    tqdm_data = tqdm.tqdm(train_data_loader2)
    for x, _ in tqdm_data:
        x0 = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
        aug0 = augmentation(x0).numpy()
        augT = ode_sampler(
            f, params, state, aug0, 25.0, 1e-3, chain_length=cfg.batch_size
        )
        xT = ode_sampler(f, params, state, x0, 25.0, 1e-3, chain_length=cfg.batch_size)
        distorg = np.linalg.norm(xT[0] - xT[1])
        distaug = np.linalg.norm(augT[0] - augT[1])
        all_distorg += distorg
        all_distaug += distaug
        num_comparisons += 1

        wandb.log(
            data={
                "Mean distance between x(T) and y(T)": float(
                    all_distorg / num_comparisons
                )
            }
        )
        wandb.log(
            data={
                "Mean distance between xaug(0) and yaug(T)": float(
                    all_distaug / num_comparisons
                )
            }
        )
        tqdm_data.set_description(
            "Mean distance between x(T) and y(T): {:5f}".format(
                all_distorg / num_comparisons
            )
        )

        tqdm_data.set_description(
            "Mean distance between x_aug(T) and y_aug(T): {:5f}".format(
                all_distaug / num_comparisons
            )
        )


if __name__ == "__main__":
    main()
