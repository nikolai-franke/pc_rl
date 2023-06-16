import pytorch_lightning as pl
from torch import Tensor


class MaskedAutoEncoder(pl.LightningModule):
    def __init__(self, embedder, encoder, decoder, prediction_head, loss_function):
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.prediction_head = prediction_head
        self.loss_function = loss_function

    def forward(self, pos: Tensor, batch: Tensor):
        x, neighborhoods, center_points = self.embedder(pos, batch)
        x_vis, mask = self.encoder(x, center_points)
        x_recovered = self.decoder(x_vis, mask, center_points)
        pos_recovered = self.prediction_head(x_recovered)

        return pos_recovered, neighborhoods, mask

    def training_step(self, data, batch_idx):
        x, neighborhoods, mask = self.forward(data.pos, data.batch)
        B, M, C = x.shape
        y = neighborhoods[~mask].reshape(B * M, -1, 3)
        loss = self.loss_function(y, x)

        return loss
