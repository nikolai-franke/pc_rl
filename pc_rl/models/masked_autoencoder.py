import pytorch_lightning as pl
import torch
from torch import Tensor

import wandb
from wandb import Object3D


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(
        self, embedder, encoder, decoder, prediction_head, loss_function, learning_rate
    ):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.prediction_head = prediction_head
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        # TODO: don't hardcode this here. Check if lightning provides a convenient way of logging each n steps
        self.log_every = 50
        self.step = 0

    def forward(self, pos: Tensor, batch: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x_vis, mask = self.encoder(x, center_points)
        x_recovered = self.decoder(x_vis, mask, center_points)
        pos_recovered = self.prediction_head(x_recovered)

        return pos_recovered, neighborhoods, mask, center_points

    def training_step(self, data, batch_idx):
        self.step += 1
        x, neighborhoods, mask, center_points = self.forward(data.pos, data.batch)
        B, M, *_ = x.shape
        _, N, *_ = neighborhoods.shape
        y = neighborhoods[~mask].reshape(B * M, -1, 3)
        x = x.reshape(B * M, -1, 3)

        with torch.no_grad():
            if self.step % self.log_every == 0:
                vis_points = neighborhoods[mask]
                input = vis_points + center_points[mask].unsqueeze(1)
                input = input.reshape(B, -1, 3)

                pred = x + center_points[~mask].unsqueeze(1)
                pred = pred.reshape(B, -1, 3)
                pred = torch.cat([input, pred], dim=1)

                ground_truth = y + center_points[~mask].unsqueeze(1)
                ground_truth = ground_truth.reshape(B, -1, 3)
                ground_truth = torch.cat([input, ground_truth], dim=1)

                wandb.log({"pred": Object3D(pred[0].cpu().numpy())})
                wandb.log({"input": Object3D(input[0].cpu().numpy())})
                wandb.log({"full": Object3D(ground_truth[0].cpu().numpy())})
                self.step = 0

        loss = self.loss_function(x, y, point_reduction="sum")[0]
        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
