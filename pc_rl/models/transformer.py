import torch
from pointnet2_ops.pointnet2_utils import Callable
from torch import nn
from torch_geometric.nn import MLP


class Attention(nn.Module):
    def __init__(
        self,
        transformer_size,
        num_heads=8,
        qkv_bias=False,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = transformer_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(transformer_size, transformer_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_rate)
        self.proj = nn.Linear(transformer_size, transformer_size)
        self.proj_drop = nn.Dropout(projection_dropout_rate)

    def forward(self, x):
        B, N, E = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, E // self.num_heads)
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

        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        transformer_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        mlp_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        mlp_activation: Callable = nn.GELU(),
        NormLayer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm_1 = NormLayer(transformer_size)
        self.norm_2 = NormLayer(transformer_size)
        mlp_hidden_dim = int(transformer_size * mlp_ratio)
        self.mlp = MLP(
            [transformer_size, mlp_hidden_dim, transformer_size],
            act=mlp_activation,
            norm=None,
            dropout=mlp_dropout_rate,
        )
        self.attention = Attention(
            transformer_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attention_dropout_rate=attention_dropout_rate,
            projection_dropout_rate=mlp_dropout_rate,
        )

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        transformer_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        attn_dropout=0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    transformer_size=transformer_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_dropout_rate=dropout,
                    attention_dropout_rate=attn_dropout,
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
        transformer_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        mlp_dropout_rate=0.0,
        attention_dropout_rate=0.0,
        NormLayer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    transformer_size=transformer_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_dropout_rate=mlp_dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = NormLayer(transformer_size)
        self.head = nn.Identity()  # TODO: maybe remove this if we don't need a head

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
        mask_ratio: float,
        depth: int,
        num_heads: int,
        embedder: Callable,
        mask_type: str = "rand",  # TODO:check if we need different mask_types
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embedder = embedder

        assert hasattr(
            self.embedder, "embedding_size"
        ), f"embedder {self.embedder} does not have attribute 'embedding_size'"

        self.embedding_size = self.embedder.embedding_size
        self.depth = depth
        self.num_heads = num_heads

        self.mask_type = mask_type

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embedding_size),
        )

        self.transformer_encoder = TransformerEncoder(
            transformer_size=self.embedding_size,
            depth=self.depth,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.embedding_size)
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
        tokens, neighborhoods, center_points = self.embedder(pos, batch)

        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center_points, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center_points, noaug=noaug)

        batch_size, _, C = tokens.shape

        x_vis = tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        masked_center = center_points[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embedding(masked_center)

        x_vis = self.transformer_encoder(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos, neighborhoods, center_points


class PointMAE(nn.Module):
    def __init__(
        self,
        transformer_dim,
        neighborhood_size,
        num_groups,
        group_size,
        decoder_depth,
        decoder_num_heads,
        loss_func,
        mask_transformer: Callable,
    ):
        self.group_size = group_size
        self.transformer_dim = transformer_dim
        self.neighborhood_size = neighborhood_size
        self.num_groups = num_groups
        self.mask_transformer = mask_transformer
        # self.mae_encoder = MaskTransformer(**mask_transformer_kwargs)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.transformer_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.transformer_dim),
        )
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        self.mae_decoder = TransformerDecoder(
            transformer_size=self.transformer_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
        )

        self.prediction_head = nn.Conv1d(self.transformer_dim, 3 * self.group_size, 1)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.loss_func = loss_func

    def forward(self, pos, batch, vis=False):
        x_vis, mask, neighborhoods, center_points = self.mask_transformer(pos, batch)

        B, _, C = x_vis.shape
        pos_emd_vis = self.decoder_pos_embed(center_points[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center_points[mask]).reshape(B, -1, C)

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

        gt_points = neighborhoods[mask].reshape(B * M, -1, 3)
        loss = self.loss_func(rebuild_points, gt_points)
        if vis:
            raise NotImplementedError
        else:
            return loss
