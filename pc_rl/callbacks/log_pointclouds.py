from __future__ import annotations

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from pc_rl.utils.logger_utils import log_point_cloud


def create_full_point_clouds(
    prediction, ground_truth, neighborhoods, mask, center_points
):
    C = neighborhoods.shape[-1]
    masked_input = neighborhoods[~mask]
    masked_input[..., :3] = masked_input[..., :3] + center_points[~mask].unsqueeze(1)
    masked_input = masked_input.reshape(-1, C)

    prediction[..., :3] = prediction[..., :3] + center_points[mask].unsqueeze(1)
    prediction = prediction.reshape(-1, C)
    prediction = torch.cat([masked_input, prediction], dim=0)

    ground_truth[..., :3] = ground_truth[..., :3] + center_points[mask].unsqueeze(1)
    ground_truth = ground_truth.reshape(-1, C)
    ground_truth = torch.cat([masked_input, ground_truth], dim=0)

    return masked_input, prediction, ground_truth


class LogPointCloudCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        B, G, C = pl_module.B, pl_module.G, pl_module.C
        ground_truth = pl_module.ground_truth.view(B, -1, G, C)  # type: ignore
        prediction = pl_module.prediction.view(B, -1, G, C)  # type: ignore

        padding_mask_without_masked_tokens = pl_module.padding_mask_without_masked_tokens  # type: ignore
        index = torch.randint(0, prediction.shape[0], (1,))
        prediction = prediction[index][~padding_mask_without_masked_tokens[index]]
        ground_truth = ground_truth[index][~padding_mask_without_masked_tokens[index]]
        center_points = pl_module.center_points
        center_points = center_points[index][~pl_module.padding_mask[index]]
        masked_input, prediction, ground_truth = create_full_point_clouds(
            prediction,
            ground_truth,
            pl_module.neighborhoods[index][~pl_module.padding_mask.reshape(B, -1)[index]],  # type: ignore
            pl_module.ae_mask[index][~pl_module.padding_mask.reshape(B, -1)[index]],  # type: ignore
            center_points,  # type: ignore
        )
        log_point_cloud("masked_input", masked_input)
        log_point_cloud("prediction", prediction)
        log_point_cloud("ground_truth", ground_truth)


class LogGPTPointCloudCallback(Callback):
    def on_validation_end(self, trainer: LightningModule, pl_module: Trainer) -> None:
        B, G, C = pl_module.B, pl_module.G, pl_module.C  # type: ignore
        prediction = pl_module.prediction.view(B, -1, G, C)  # type: ignore
        ground_truth = pl_module.ground_truth.view(B, -1, G, C)  # type: ignore
        center_points = pl_module.center_points  # type: ignore
        padding_mask = pl_module.padding_mask  # type: ignore

        # only log one point cloud per batch
        index = torch.randint(0, prediction.shape[0], (1,))  # type: ignore
        ground_truth = ground_truth[index][~padding_mask[index]]
        prediction = prediction[index][~padding_mask[index]]
        center_points = center_points[index][~padding_mask[index]]

        ground_truth[..., :3] = ground_truth[..., :3] + center_points.unsqueeze(1)
        prediction[..., :3] = prediction[..., :3] + center_points.unsqueeze(1)

        log_point_cloud("prediction", prediction.reshape(-1, C))
        log_point_cloud("ground_truth", ground_truth.reshape(-1, C))
