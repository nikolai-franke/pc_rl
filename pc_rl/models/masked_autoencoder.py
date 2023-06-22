import pytorch_lightning as pl
import torch
from torch import Tensor


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

    def forward(self, pos: Tensor, batch: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x_vis, mask = self.encoder(x, center_points)
        x_recovered, padding_mask = self.decoder(x_vis, mask, center_points)
        pos_recovered = self.prediction_head(x_recovered)

        return pos_recovered, neighborhoods, mask, padding_mask, center_points

    def training_step(self, data, batch_idx):
        prediction, neighborhoods, mask, padding_mask, _ = self.forward(
            data.pos, data.batch
        )
        B, M, G, _ = prediction.shape
        padding_mask = padding_mask.unsqueeze(-2).expand((-1, -1, G, -1))
        padding_mask = padding_mask[mask]
        ground_truth = neighborhoods[mask].reshape(B * M, -1, 3)

        prediction = prediction.reshape(B * M, -1, 3)
        prediction[padding_mask] = 0.0
        ground_truth[padding_mask] = 0.0

        loss = self.loss_function(prediction, ground_truth, point_reduction="sum")[0]
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
