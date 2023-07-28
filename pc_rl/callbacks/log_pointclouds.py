import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from pc_rl.utils.logger_utils import log_point_cloud


def create_full_point_clouds(x, y, neighborhoods, mask, center_points):
    masked_input = neighborhoods[~mask]
    masked_input = masked_input + center_points[~mask].unsqueeze(1)
    masked_input = masked_input.reshape(-1, 3)

    prediction = x + center_points[mask].unsqueeze(1)
    prediction = prediction.reshape(-1, 3)
    prediction = torch.cat([masked_input, prediction], dim=0)

    ground_truth = y + center_points[mask].unsqueeze(1)
    ground_truth = ground_truth.reshape(-1, 3)
    ground_truth = torch.cat([masked_input, ground_truth], dim=0)

    return masked_input, prediction, ground_truth


class LogPointCloudCallback(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        B, _, G, _ = pl_module.test_neighborhoods.shape  # type: ignore
        ground_truth = pl_module.test_ground_truth.reshape(B, -1, G, 3)  # type: ignore
        prediction = pl_module.test_prediction.reshape(B, -1, G, 3)  # type: ignore
        index = torch.randint(0, prediction.shape[0], (1,))
        masked_input, prediction, ground_truth = create_full_point_clouds(
            prediction[index],
            ground_truth[index],
            pl_module.test_neighborhoods[index],  # type: ignore
            pl_module.test_mask[index],  # type: ignore
            pl_module.test_center_points[index],  # type: ignore
        )
        log_point_cloud("masked_input", masked_input)
        log_point_cloud("prediction", prediction)
        log_point_cloud("ground_truth", ground_truth)
