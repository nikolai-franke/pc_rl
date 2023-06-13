import torch
from torch import nn
from torch_geometric.nn import MLP

from pc_rl.models.embedder import Embedder


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        attn_dropout=0.0,
        act_cls=nn.GELU,
        NormLayer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = NormLayer(dim)
        self.norm2 = NormLayer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            [dim, mlp_hidden_dim, dim],
            act=act_cls(),
            norm=None,
            dropout=dropout,
        )
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        attn_dropout=0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        attn_dropout=0.0,
        NormLayer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = NormLayer(embed_dim)
        self.head = nn.Identity()  # TODO: why do we need this?

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for block in self.blocks:
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))
        return x


class MaskTransformer(nn.Module):
    def __init__(
        self,
        mask_ratio=0.6,
        transformer_dim=384,
        depth=12,
        num_heads=6,
        mask_type="rand",
        embedder_kwargs={},
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.transformer_dim = transformer_dim
        self.depth = depth
        self.num_heads = num_heads

        self.encoder = Embedder(**embedder_kwargs)
        self.mask_type = mask_type

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim),
        )

        self.blocks = TransformerEncoder(
            embed_dim=self.transformer_dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.transformer_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2].bool())
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = torch.randint(points.shape[1] - 1, (1,))
            # index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_masks = int(self.mask_ratio * G)

        # overall_mask = np.zeros([B, G])
        overall_mask = torch.zeros([B, G])
        for i in range(B):
            mask = torch.hstack(
                [
                    torch.zeros(G - self.num_masks),
                    torch.ones(self.num_masks),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            overall_mask[i, :] = mask
            overall_mask = overall_mask.bool()

        return overall_mask.to(center.device)

    def forward(self, pos, batch, noaug=False):
        group_input_tokens, neighborhood, center = self.encoder(pos, batch)

        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        batch_size, _, C = group_input_tokens.shape

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embedding(masked_center)

        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos, neighborhood, center


class PointMAE(nn.Module):
    def __init__(
        self,
        transformer_dim,
        neighborhood_size,
        num_groups,
        group_size,
        mask_transformer_kwargs,
        decoder_depth,
        decoder_num_heads,
        loss_func,
    ):
        self.group_size = group_size
        self.transformer_dim = transformer_dim
        self.neighborhood_size = neighborhood_size
        self.num_groups = num_groups
        self.mae_encoder = MaskTransformer(**mask_transformer_kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim),
        )
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        self.mae_decoder = TransformerDecoder(
            embed_dim=self.transformer_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
        )

        self.prediction_head = nn.Conv1d(self.transformer_dim, 3 * self.group_size, 1)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.loss_func = loss_func

    def forward(self, pos, batch, vis=False):
        x_vis, mask, neighborhood, center = self.mae_encoder(pos, batch)

        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.mae_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape

        rebuild_points = (
            self.prediction_head(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss = self.loss_func(rebuild_points, gt_points)
        if vis:
            raise NotImplementedError
        else:
            return loss
