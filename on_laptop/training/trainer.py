from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Dict, List
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


def save_checkpoint(modules: TrainModules, epoch: int, ckpt_path: Path) -> None:
    save_dict = {
        "epoch": epoch,
        "model": modules.model.state_dict(),
        "optimizer": modules.optimizer.state_dict(),
        "scheduler": modules.scheduler.state_dict(),
    }
    torch.save(save_dict, ckpt_path)


def resume_checkpoint(modules: TrainModules, ckpt_path: Path) -> int:
    """Return epoch"""
    resume_dict = torch.load(ckpt_path)
    modules.model.load_state_dict(resume_dict["model"])
    modules.optimizer.load_state_dict(resume_dict["optimizer"])
    modules.scheduler.load_state_dict(resume_dict["scheduler"])
    return resume_dict["epoch"]


def calc_accuracy(pred_yaw: Tensor, truth_yaw: Tensor) -> Dict[str, float]:
    if truth_yaw.shape[-1] == 1:
        return {"mse": (truth_yaw - pred_yaw).pow(2).mean().item()}

    pred_logits = pred_yaw.argmax(dim=-1)
    accuracy = (pred_logits == truth_yaw).sum() / truth_yaw.nelement()
    # cls/10 = angle radians
    mse = ((truth_yaw - pred_logits) / 10).pow(2).mean().item()
    return {"accuracy": accuracy, "mse": mse}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Dict[str, nn.Module],
    optimizer: Optimizer,
    logger: SummaryWriter,
    epoch: int,
    max_epoch: int,
) -> None:

    with tqdm(
        total=len(dataloader), desc=f"Training [{epoch:03}|{max_epoch:03}]"
    ) as pbar:
        for sample in dataloader:
            optimizer.zero_grad()
            preds = model(sample)

            total_loss = torch.zeros(1)
            for crit in criterion:
                loss: Tensor = criterion[crit](preds, sample["yaw"])
                logger.add_scalar(
                    f"train/{crit}", loss.item(), pbar.n + len(dataloader) * epoch
                )
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                stats = calc_accuracy(preds, sample["yaw"])
                for name, value in stats.items():
                    logger.add_scalar(
                        f"train/{name}", value, pbar.n + len(dataloader) * epoch
                    )

            pbar.update(1)


@inference_mode()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    logger: SummaryWriter,
    epoch: int,
    max_epoch: int,
) -> None:
    # Statistic accumulators
    acc_samples: Dict[str, List[float]] = {}
    loss_samples: Dict[str, List[float]] = {c: [] for c in criterion}

    with tqdm(
        total=len(dataloader), desc=f"Validating [{epoch:03}|{max_epoch:03}]"
    ) as pbar:
        for sample in dataloader:
            preds = model(sample)
            for crit in criterion:
                loss: Tensor = criterion[crit](preds, sample["yaw"])
                loss_samples[crit].append(loss.item())

            stats = calc_accuracy(preds, sample["yaw"])
            for name, value in stats.items():
                if name not in acc_samples:
                    acc_samples[name] = []
                acc_samples[name].append(value)

            pbar.update(1)

    # Log statistics
    for stat, samples in acc_samples.items():
        logger.add_scalar(
            f"validate/{stat}", sum(samples) / len(samples), len(dataloader) * epoch
        )
    for crit, samples in loss_samples.items():
        logger.add_scalar(
            f"validate/{crit}", sum(samples) / len(samples), len(dataloader) * epoch
        )


def train(modules: TrainModules, epochs: int, root_path: Path):
    """Train model to target epoch, save checkpoints to root_path"""
    logger = SummaryWriter(root_path)

    ckpt_path = root_path / "latest.pth"
    start_epoch = resume_checkpoint(modules, ckpt_path) if ckpt_path.exists() else 0

    for epoch in range(start_epoch, epochs):
        train_epoch(
            modules.model,
            modules.train_loader,
            modules.criterion,
            modules.optimizer,
            logger,
            epoch,
            epochs,
        )

        validate_epoch(
            modules.model,
            modules.val_loader,
            modules.criterion,
            logger,
            epoch + 1,
            epochs,
        )

        save_checkpoint(modules, epoch + 1, ckpt_path)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-x", "--experiment", type=Path, help="Path to experiment folder"
    )
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs to train")
    parser.add_argument(
        "-w", "--workers", default=0, type=int, help="Dataloader Workers"
    )
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
    config["dataloader"]["workers"] = args.workers
    train_modules = initialize_modules(config)
    train(train_modules, args.epochs, exp_path)


if __name__ == "__main__":
    main()
