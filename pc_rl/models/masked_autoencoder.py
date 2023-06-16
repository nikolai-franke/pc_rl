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
        x_recovered = self.decoder(x_vis, mask, center_points)
        pos_recovered = self.prediction_head(x_recovered)

        return pos_recovered, neighborhoods, mask

    def training_step(self, data, batch_idx):
        x, neighborhoods, mask = self.forward(data.pos, data.batch)
        B, M, *_ = x.shape
        y = neighborhoods[~mask].reshape(B * M, -1, 3)
        x = x.reshape(B * M, -1, 3)

        loss = self.loss_function(y, x)
        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.05
        )
