import torch
from torch import nn
from torch_geometric.nn import MLP


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


class MaskTransfromer(nn.Module):
    def __init__(
        self,
        mask_ratio=0.6,
        transformer_dim=384,
        depth=12,
        num_heads=6,
        encoder_dim=384,
        mask_type="rand",
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.transformer_dim = transformer_dim
        self.depth = depth
        self.num_heads = num_heads
        self.encoder_dim = encoder_dim
        self.mask_type = mask_type

        self.encoder = Encoder

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
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2].bool())
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]
            mask_num = int(self.mask_ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx.to(center.device))

    def _mask_center_rand(self, center, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_masks = int(self.mask_ratio * G)

        for i in range(B):
            mask = np.hstack(
                [
                    np.zeros(G - self.num_mask),
                    np.ones(self.num_mask),
                ]
            )
            np.random.shuffle(mask)
            overall_mask[i, :] = mask

        overall_mask = torch.from_numpy(overall_mask).bool()

        return overall_mask.to(center.device)

    def forward(self, neighborhood, center, noaug=False):
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)
        batch_size, seq_len, C = group_input_tokens.size()
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embedding(masked_center)

        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
