from typing import Dict

from torch import nn


def get_criterion(config) -> Dict[str, nn.Module]:
    avail_losses = {"MSE": nn.MSELoss}

    losses = {}
    for loss in config["criterion"]:
        losses[loss["type"]] = avail_losses[loss["type"]](**loss["args"])

    return losses
