from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Dict
import yaml

import torch
from torch import nn, Tensor, inference_mode
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from model import get_model
from dataloader import get_dataloader, DataLoader
from optimizer import _LRScheduler, Optimizer, get_optimizer, get_scheduler
from criterion import get_criterion


@dataclass
class TrainModules:
    model: nn.Module
    criterion: Dict[str, nn.Module]
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: Optimizer
    scheduler: _LRScheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Dict[str, nn.Module],
    optimizer: Optimizer,
    logger: SummaryWriter,
    epoch: int,
) -> None:

    with tqdm(total=len(dataloader), desc="Training") as pbar:
        for image, label in dataloader:
            optimizer.zero_grad()
            out = model(image)

            total_loss = torch.zeros(1)
            for crit in criterion:
                loss: Tensor = criterion[crit](label, out)
                logger.add_scalar(
                    f"train/{crit}", loss.item(), pbar.n + len(dataloader) * epoch
                )
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            pbar.update(1)


@inference_mode()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    logger: SummaryWriter,
    epoch: int,
) -> None:
    with tqdm(total=len(dataloader), desc="Validating") as pbar:
        for image, label in dataloader:
            out = model(image)
            for crit in criterion:
                loss: Tensor = criterion[crit](label, out)
                logger.add_scalar(
                    f"validate/{crit}", loss.item(), pbar.n + len(dataloader) * epoch
                )
            pbar.update(1)


def train(modules: TrainModules, epochs: int, root_path: Path):
    logger = SummaryWriter(root_path)

    for _ in range(epochs):
        train_epoch(
            modules.model,
            modules.train_loader,
            modules.criterion,
            modules.optimizer,
        )

        validate_epoch(
            modules.model,
            modules.val_loader,
            modules.criterion,
        )


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-x", "--experiment", type=Path, help="Path to experiment folder"
    )
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    return parser.parse_args()


def initialize_modules(config) -> TrainModules:
    model = get_model(config)
    train_l, val_l = get_dataloader(config)
    optim = get_optimizer(config, model)
    scheduler = get_scheduler(config, optim)
    criterion = get_criterion(config)
    return TrainModules(model, criterion, train_l, val_l, optim, scheduler)


def main() -> None:
    args = get_args()
    exp_path: Path = args.experiment
    with open(exp_path / "config.yml", "r") as f:
        config = yaml.safe_load(f)
    train_modules = initialize_modules(config)
    train(train_modules, args.epochs, exp_path)


if __name__ == "__main__":
    main()
