import torch
import torch.nn as nn

from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.models.modules.masked_decoder import MaskedDecoder, MaePredictionHead



class AuxMae(nn.Module):
    def __init__(
        self,
        masked_encoder: MaskedEncoder,
        masked_decoder: MaskedDecoder,
        mae_prediction_head: MaePredictionHead,
        mlp_head: nn.Module,
    ):
        super().__init__()
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.mlp_head = mlp_head
        self.dim = self.masked_encoder.dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))

    def forward(self, x, center_points, neighborhoods):
        # MAE part
        x_vis, ae_mask = self.masked_encoder(x, center_points)
        pos_recovered, padding_mask = self.masked_decoder(x_vis, ae_mask, center_points)
        pos_recovered = self.mae_prediction_head(pos_recovered)
        B, M, G, _ = pos_recovered.shape
        padding_mask = padding_mask.view(B, -1, 1, 1).expand(-1, -1, G, 3)
        padding_mask = padding_mask[ae_mask]
        pos_ground_truth = neighborhoods[ae_mask].reshape(B * M, -1, 3)
        pos_prediction = pos_recovered.reshape(B * M, -1, 3)
        pos_prediction[padding_mask] = 0.0
        pos_ground_truth[padding_mask] = 0.0

        # classification part
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(x.shape[0], -1, -1)
        # use pos_embedder, transformer_encoder, and norm from masked_encoder
        pos = self.masked_encoder.pos_embedder(center_points)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.masked_encoder.transformer_encoder(x, pos)
        x = self.masked_encoder.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        x_out = self.mlp_head(concat_f)


        return x_out, pos_prediction, pos_ground_truth
