import torch
from torch import nn

from pc_rl.models.modules.mae import MaskedDecoder, MaskedEncoder


class AuxiliaryMae(nn.Module):
    def __init__(
        self,
        masked_encoder: MaskedEncoder,
        masked_decoder: MaskedDecoder,
        mae_prediction_head,
        mlp_head,
    ):
        self.masked_encoder = masked_encoder
        self.masked_decoder = masked_decoder
        self.mae_prediction_head = mae_prediction_head
        self.mlp_head = mlp_head
        self.dim = self.masked_encoder.embedding_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.dim))

    def forward(self, x, center_points):
        x_vis, ae_mask = self.masked_encoder(x, center_points)
        x_recovered, padding_mask = self.masked_decoder(x_vis, ae_mask, center_points)
        pos_recovered = self.mae_prediction_head(x_recovered)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(x.shape[0], -1, -1)
        # use pos_embedder and transformer_encoder from masked_encoder
        pos = self.masked_encoder.pos_embedder(center_points)
        pos = torch.cat((cls_pos, pos), dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.masked_encoder.transformer_encoder(x)
        x = self.masked_encoder.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1].max(1)[0]], dim=-1)
        out = self.mlp_head(concat_f)

        return pos_recovered, out


