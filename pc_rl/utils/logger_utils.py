from __future__ import annotations

from torch import Tensor

import wandb
from wandb import Object3D


def log_point_cloud(key: str, point_cloud: Tensor):
    if point_cloud.shape[-1] == 6:
        point_cloud[..., 3:] *= 255
    wandb.log({key: Object3D(point_cloud.cpu().numpy())})
