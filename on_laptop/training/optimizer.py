from typing import Dict

from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler
from torch.optim import SGD, Optimizer


def get_optimizer(config, model: nn.Module) -> Optimizer:
    optims: Dict[str, Optimizer] = {"SGD": SGD}
    optim = optims[config["optimizer"]["type"]](
        model.parameters(), **config["optimizer"]["args"]
    )
    return optim


def get_scheduler(config, optim) -> _LRScheduler:
    schedulers = {
        "ExponentialLR": ExponentialLR,
    }
    sched_cfg = config["scheduler"]
    scheduler = schedulers[sched_cfg["type"]](optim, *sched_cfg["args"])
    return scheduler
