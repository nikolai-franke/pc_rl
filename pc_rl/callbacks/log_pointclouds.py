import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from pc_rl.logger_utils import log_point_cloud


def create_full_point_clouds(x, y, B, neighborhoods, mask, center_points):
    masked_input = neighborhoods[mask]
    masked_input = masked_input + center_points[mask].unsqueeze(1)
    masked_input = masked_input.reshape(B, -1, 3)

    prediction = x + center_points[~mask].unsqueeze(1)
    prediction = prediction.reshape(B, -1, 3)
    prediction = torch.cat([masked_input, prediction], dim=1)

    ground_truth = y + center_points[~mask].unsqueeze(1)
    ground_truth = ground_truth.reshape(B, -1, 3)
    ground_truth = torch.cat([masked_input, ground_truth], dim=1)

    return masked_input, prediction, ground_truth


class LogPointCloudCallback(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # TODO: this should be changed later, we can just use a Subset of the validation dataset
        with torch.no_grad():
            data = next(iter(trainer.train_dataloader)).cuda()  # type: ignore
            x, neighborhoods, mask, center_points = pl_module.forward(
                data.pos, data.batch
            )
            B, M, *_ = x.shape
            y = neighborhoods[~mask].reshape(B * M, -1, 3)
            x = x.reshape(B * M, -1, 3)
            masked_input, prediction, ground_truth = create_full_point_clouds(
                x, y, B, neighborhoods, mask, center_points
            )
            log_point_cloud("masked_input", masked_input[0])
            log_point_cloud("prediction", prediction[0])
            log_point_cloud("ground_truth", ground_truth[0])
