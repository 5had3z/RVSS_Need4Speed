from typing import Dict

import torch
from torch import nn


def get_criterion(config) -> Dict[str, nn.Module]:
    avail_losses = {
        "MSE": nn.MSELoss,
        "L1": nn.L1Loss,
        "CrossEntropy": nn.CrossEntropyLoss,
    }

    losses = {}
    for loss in config["criterion"]:
        if "weight" in loss["args"]:
            loss["args"]["weight"] = torch.tensor(loss["args"]["weight"])
        losses[loss["type"]] = avail_losses[loss["type"]](**loss["args"])

    return losses
