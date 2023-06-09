from torch import Tensor

import wandb
from wandb import Object3D


def log_point_cloud(key: str, point_cloud: Tensor):
    wandb.log({key: Object3D(point_cloud.cpu().numpy())})
