from __future__ import annotations

from typing import Literal

import torch
from geomloss import SamplesLoss

from pc_rl.utils.chamfer import chamfer_distance


def get_loss_fn(name: Literal["chamfer", "sinkhorn"], loss_kwargs: dict = {}):
    if name == "chamfer":

        def loss_fn(prediction, ground_truth):  # type: ignore
            return chamfer_distance(prediction, ground_truth, **loss_kwargs)[0]

    elif name == "sinkhorn":
        sinkhorn = SamplesLoss("sinkhorn", **loss_kwargs)

        def loss_fn(prediction, ground_truth):
            return torch.mean(sinkhorn(prediction, ground_truth))

    else:
        raise ValueError(f"Invalid name: {name}")

    return loss_fn
