import torch
import torch.nn as nn
import torch_geometric
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from torch_geometric.nn import MLP

from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.prediction_head import MaePredictionHead
from pc_rl.models.modules.transformer import Attention as NewAttention
from pc_rl.models.modules.transformer import Block as NewBlock
from pc_rl.models.modules.transformer import MaskedDecoder as NewMaskedDecoder
from pc_rl.models.modules.transformer import MaskedEncoder as NewMaskedEncoder
from pc_rl.models.modules.transformer import \
    TransformerDecoder as NewTransformerDecoder
from pc_rl.models.modules.transformer import \
    TransformerEncoder as NewTransformerEncoder


def custom_fps(data, number):
    """
    data B N 3
    number int
    """
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx)
        .transpose(1, 2)  # type: ignore
        .contiguous()
    )
    return fps_data


class OldEncoder(nn.Module):  ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class OldGroup(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = custom_fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class OldMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OldAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        bias=False,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=bias)
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
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OldBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # TODO: Check if DropPath is useful (I think it isn't)
        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = OldMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = OldAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))
        return x


class OldTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                OldBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class OldTransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                OldBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

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
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(
            self.norm(x[:, -return_token_num:])
        )  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(
        self,
        mask_ratio,
        trans_dim,
        depth,
        drop_path_rate,
        num_heads,
        encoder_dims,
        mask_type,
    ):
        super().__init__()
        # define the transformer argparse
        self.mask_ratio = mask_ratio
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.num_heads = num_heads
        # print_log(f"[args] {config.transformer_config}", logger="Transformer")
        # embedding
        self.encoder_dims = encoder_dims
        self.encoder = OldEncoder(encoder_channel=self.encoder_dims)

        self.mask_type = mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = OldTransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
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
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            # index = random.randint(0, points.size(1) - 1)
            index = torch.randint(points.shape[1] - 1, (1,))

            distance_matrix = torch.norm(
                points[:, index].reshape(1, 1, 3) - points, p=2, dim=-1
            )  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        """
        center : B G 3
        --------------
        mask : B G (bool)
        """
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        # overall_mask = np.zeros([B, G])
        # for i in range(B):
        #     mask = np.hstack(
        #         [
        #             np.zeros(G - self.num_mask),
        #             np.ones(self.num_mask),
        #         ]
        #     )
        #     np.random.shuffle(mask)
        #     overall_mask[i, :] = mask
        # overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        overall_mask = torch.zeros([B, G])
        for i in range(B):
            mask = torch.hstack(
                [
                    torch.zeros(G - self.num_mask),
                    torch.ones(self.num_mask),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            overall_mask[i, :] = mask
            overall_mask = overall_mask.bool()

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        # generate mask
        if self.mask_type == "rand":
            bool_masked_pos = self._mask_center_rand(center, noaug=noaug)  # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class Point_MAE(nn.Module):
    def __init__(
        self, trans_dim, mae_encoder, mae_decoder, group_size, num_group, drop_path_rate
    ):
        super().__init__()
        # print_log(f"[Point_MAE] ", logger="Point_MAE")
        # self.config = config
        # self.trans_dim = config.transformer_config.trans_dim
        self.trans_dim = trans_dim
        # self.MAE_encoder = MaskTransformer(config)
        self.MAE_encoder = mae_encoder
        # self.group_size = config.group_size
        self.group_size = group_size
        # self.num_group = config.num_group
        self.num_group = num_group
        # self.drop_path_rate = config.transformer_config.drop_path_rate
        self.drop_path_rate = drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        # self.decoder_depth = config.transformer_config.decoder_depth
        # self.decoder_num_heads = config.transformer_config.decoder_num_heads
        self.MAE_decoder = mae_decoder
        # dpr = [
        #     x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)
        # ]
        # self.MAE_decoder = TransformerDecoder(
        #     embed_dim=self.trans_dim,
        #     depth=self.decoder_depth,
        #     drop_path_rate=dpr,
        #     num_heads=self.decoder_num_heads,
        # )

        # print_log(
        #     f"[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...",
        #     logger="Point_MAE",
        # )
        self.group_divider = OldGroup(
            num_group=self.num_group, group_size=self.group_size
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # self.loss = config.loss
        # loss
        # self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == "cdl2":
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = (
            self.increase_dim(x_rec.transpose(1, 2))
            .transpose(1, 2)
            .reshape(B * M, -1, 3)
        )  # B M 1024

        # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        # loss1 = self.loss_func(rebuild_points, gt_points)

        # if vis:  # visualization
        #     vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
        #     full_vis = vis_points + center[~mask].unsqueeze(1)
        #     full_rebuild = rebuild_points + center[mask].unsqueeze(1)
        #     full = torch.cat([full_vis, full_rebuild], dim=0)
        #     # full_points = torch.cat([rebuild_points,vis_points], dim=0)
        #     full_center = torch.cat([center[mask], center[~mask]], dim=0)
        #     # full = full_points + full_center.unsqueeze(1)
        #     ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
        #     ret1 = full.reshape(-1, 3).unsqueeze(0)
        #     # return ret1, ret2
        #     return ret1, ret2, full_center
        # else:
        # return loss1
        return rebuild_points


###################################################################################################
############################################TESTS##################################################
###################################################################################################


class TestPointMAE:
    device = "cuda"
    num_points_per_batch = 1024
    num_batches = 16
    num_groups = 20
    group_size = 3
    embedding_size = 1024
    # seed = random.randint(0, 10**6)
    seed = 0
    mlp_ratio = 4.0
    sampling_ratio = num_groups / num_points_per_batch

    mask_ratio = 0.6
    transformer_dim = 1024
    transformer_depth = 4
    drop_path_rate = 0.0
    num_heads = 8
    mask_type = "rand"
    # mask_type = "asdf"

    embedder_mlp_1 = MLP([3, 128, 256])
    embedder_mlp_2 = MLP([512, 512, embedding_size])
    embedder = Embedder(
        mlp_1=embedder_mlp_1,
        mlp_2=embedder_mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        random_start=False,
    )

    def init_layers(self, layers):
        for layer in layers:
            if (
                isinstance(layer, torch.nn.Conv1d)
                or isinstance(layer, torch_geometric.nn.dense.linear.Linear)
                or isinstance(layer, torch.nn.Linear)
                or isinstance(layer, torch.nn.LayerNorm)
            ):
                torch.manual_seed(self.seed)
                torch.nn.init.uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.uniform_(layer.bias)  # type: ignore
            elif isinstance(layer, torch.nn.Parameter):
                torch.manual_seed(self.seed)
                torch.nn.init.uniform_(layer)

    def test_embedder(self):
        old_group = OldGroup(self.num_groups, self.group_size).to(self.device)
        old_embedder = OldEncoder(self.embedding_size).to(self.device)
        new_embedder = self.embedder.to(self.device)
        self.init_layers(old_embedder.modules())
        self.init_layers(new_embedder.modules())
        old_input_tensor = torch.rand(
            self.num_batches, self.num_points_per_batch, 3
        ).to(self.device)
        old_neighborhood, old_center = old_group.forward(old_input_tensor)
        old_out = old_embedder.forward(old_neighborhood)

        new_input_tensor = old_input_tensor.reshape(
            self.num_batches * self.num_points_per_batch, -1
        ).to(self.device)
        batch_tensor = torch.arange(self.num_batches, dtype=torch.long)
        batch_tensor = batch_tensor.repeat_interleave(self.num_points_per_batch).to(
            self.device
        )
        new_out, new_neighborhood, new_center = new_embedder.forward(
            new_input_tensor, batch_tensor
        )

        assert torch.allclose(old_out, new_out, rtol=1e-4, atol=1e-4)
        # assert torch.allclose(old_out, new_out)
        assert torch.equal(old_neighborhood, new_neighborhood)
        assert torch.equal(old_center, new_center)

    def test_attention(self):
        old_attention = OldAttention(
            dim=self.embedding_size,
            num_heads=self.num_heads,
        ).to(self.device)
        new_attention = nn.MultiheadAttention(
            self.embedding_size, self.num_heads, batch_first=True, bias=False
        ).to(self.device)
        self.init_layers(old_attention.modules())
        self.init_layers(new_attention.modules())
        self.init_layers(new_attention.parameters())
        input_tensor = torch.rand(
            self.num_batches, self.num_points_per_batch, self.embedding_size
        ).to(self.device)
        old_out = old_attention(input_tensor)
        new_out = new_attention(
            input_tensor, input_tensor, input_tensor, need_weights=False
        )[0]

        print(old_out - new_out)
        assert torch.allclose(old_out, new_out)

    def test_block(self):
        old_block = OldBlock(
            dim=self.embedding_size, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio
        ).to(self.device)

        new_attention = nn.MultiheadAttention(
            self.embedding_size, self.num_heads, batch_first=True, bias=False
        ).to(self.device)
        mlp = MLP(
            [
                self.embedding_size,
                int(self.embedding_size * self.mlp_ratio),
                self.embedding_size,
            ],
            act="gelu",
            norm=None,
            dropout=0.0,
        )
        new_block = NewBlock(attention=new_attention, mlp=mlp).to(self.device)

        self.init_layers(old_block.modules())

        self.init_layers(new_block.modules())
        self.init_layers(new_attention.modules())
        self.init_layers(new_attention.parameters())

        input_tensor = torch.rand(self.num_batches, 1024, self.embedding_size).to(
            self.device
        )
        old_out = old_block.forward(input_tensor)
        new_out = new_block.forward(input_tensor)
        print(old_out - new_out)
        # assert torch.allclose(old_out, new_out)

    def test_transformer_encoder(self):
        old_encoder = OldTransformerEncoder(
            embed_dim=self.transformer_dim,
            depth=self.transformer_depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        ).to(self.device)
        encoder_layer = TransformerEncoderLayer(
            dim_feedforward=int(self.mlp_ratio * self.embedding_size),
            d_model=self.transformer_dim,
            nhead=self.num_heads,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        new_encoder = PtTransformerEncoder(
            encoder_layer,
            num_layers=self.transformer_depth,
        ).to(self.device)
        # new_encoder = self.encoder.to(self.device)
        self.init_layers(old_encoder.modules())
        self.init_layers(new_encoder.modules())

        input_x = torch.rand(
            self.num_batches, self.num_groups, self.transformer_dim
        ).to(self.device)
        input_pos = torch.rand(
            self.num_batches, self.num_groups, self.transformer_dim
        ).to(self.device)

        old_out = old_encoder.forward(input_x, input_pos)
        new_out = new_encoder.forward(input_x, input_pos)

        assert torch.allclose(old_out, new_out, rtol=1e-4, atol=1e-4)

    def test_transformer_decoder(self):
        return_token_num = 10
        torch.manual_seed(self.seed)
        old_decoder = OldTransformerDecoder(
            self.transformer_dim, self.transformer_depth, self.num_heads, self.mlp_ratio
        ).to(self.device)
        self.init_layers(old_decoder.modules())
        torch.manual_seed(self.seed)
        new_decoder = NewTransformerDecoder(blocks=self.blocks).to(self.device)
        self.init_layers(new_decoder.modules())

        input_x = torch.rand(
            self.num_batches, self.num_groups, self.transformer_dim
        ).to(self.device)
        input_pos = torch.rand(
            self.num_batches, self.num_groups, self.transformer_dim
        ).to(self.device)

        old_out = old_decoder.forward(input_x, input_pos, return_token_num)
        new_out = new_decoder.forward(input_x, input_pos, return_token_num)

        assert torch.allclose(old_out, new_out)

    def test_masked_encoder(self):
        torch.manual_seed(self.seed)
        old_mask_transformer = MaskTransformer(
            mask_ratio=self.mask_ratio,
            trans_dim=self.transformer_dim,
            depth=self.transformer_depth,
            drop_path_rate=self.drop_path_rate,
            num_heads=self.num_heads,
            encoder_dims=self.embedding_size,
            mask_type=self.mask_type,
        ).to(self.device)

        torch.manual_seed(self.seed)
        new_mask_transformer = self.masked_encoder.to(self.device)
        torch.manual_seed(self.seed)
        self.init_layers(old_mask_transformer.modules())

        torch.manual_seed(self.seed)
        self.init_layers(new_mask_transformer.modules())

        old_input_tensor = torch.rand(
            self.num_batches, self.num_points_per_batch, 3
        ).to(self.device)
        new_input_tensor = old_input_tensor.reshape(
            self.num_batches * self.num_points_per_batch, -1
        ).to(self.device)
        batch_tensor = torch.arange(self.num_batches, dtype=torch.long)
        batch_tensor = batch_tensor.repeat_interleave(self.num_points_per_batch).to(
            self.device
        )
        group = OldGroup(self.num_groups, self.group_size).to(self.device)
        neighborhood, center = group.forward(old_input_tensor)
        torch.manual_seed(self.seed)
        old_x_vis, old_mask = old_mask_transformer.forward(neighborhood, center)
        torch.manual_seed(self.seed)

        x, _, center_points = self.embedder(new_input_tensor, batch_tensor)

        new_x_vis, new_mask = new_mask_transformer.forward(x, center_points)

        # print(torch.sum(torch.abs(diff := (old_x_vis - new_x_vis))), diff.shape)
        assert torch.allclose(old_x_vis, new_x_vis, rtol=1e-4, atol=1e-4)
        assert torch.equal(old_mask, ~new_mask)

    def test_point_mae(self):
        old_encoder = MaskTransformer(
            mask_ratio=self.mask_ratio,
            trans_dim=self.transformer_dim,
            depth=self.transformer_depth,
            num_heads=self.num_heads,
            drop_path_rate=self.drop_path_rate,
            encoder_dims=self.embedding_size,
            mask_type=self.mask_type,
        )
        old_decoder = OldTransformerDecoder(
            embed_dim=self.transformer_dim,
            depth=self.transformer_depth,
            drop_path_rate=self.drop_path_rate,
            num_heads=self.num_heads,
        )
        old_point_mae = Point_MAE(
            self.transformer_dim,
            mae_encoder=old_encoder,
            mae_decoder=old_decoder,
            group_size=self.group_size,
            num_group=self.num_groups,
            drop_path_rate=self.drop_path_rate,
        ).to(self.device)

        old_input_tensor = torch.rand(
            self.num_batches, self.num_points_per_batch, 3
        ).to(self.device)
        new_input_tensor = old_input_tensor.reshape(
            self.num_batches * self.num_points_per_batch, -1
        ).to(self.device)
        batch_tensor = torch.arange(self.num_batches, dtype=torch.long)
        batch_tensor = batch_tensor.repeat_interleave(self.num_points_per_batch).to(
            self.device
        )
        self.embedder.to(self.device)
        self.masked_encoder.to(self.device)
        self.masked_decoder.to(self.device)
        self.prediction_head.to(self.device)
        self.init_layers(old_point_mae.modules())
        self.init_layers(self.embedder.modules())
        self.init_layers(self.masked_encoder.modules())
        self.init_layers(self.masked_decoder.modules())
        self.init_layers(self.prediction_head.modules())

        x, _, center_points = self.embedder.forward(new_input_tensor, batch_tensor)
        x_vis, mask = self.masked_encoder.forward(x, center_points)
        decoder_out = self.masked_decoder.forward(x_vis, mask, center_points)
        new_out = self.prediction_head.forward(decoder_out)

        old_out = old_point_mae.forward(old_input_tensor)
        assert torch.allclose(old_out, new_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test = TestPointMAE()
    # test.test_embedder()
    # test.test_attention()
    test.test_block()
    # test.test_transformer_encoder()
    # test.test_transformer_decoder()
    # test.test_point_mae()
