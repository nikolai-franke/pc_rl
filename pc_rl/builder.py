from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from parllel.replays import BatchedDataLoader
from parllel.torch.agents.agent import TorchAgent
from parllel.torch.algos.ppo import PPO, SamplesForLoss
from parllel.torch.models import MlpModel, Optional
from torch.nn import MultiheadAttention
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.nn import MLP

from pc_rl.models.aux_mae import AuxMae
from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.mae import MaskedAutoEncoder
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.models.modules.masked_encoder import MaskedEncoder
from pc_rl.models.modules.transformer import (TransformerBlock,
                                              TransformerDecoder,
                                              TransformerEncoder)
from pc_rl.models.pg.aux_mae_categorical import AuxMaeCategoricalPgModel
from pc_rl.models.pg.finetune_categorical import CategoricalPgModel
from pc_rl.models.pg.finetune_continuous import ContinuousPgModel
from pc_rl.algos.aux_ppo import AuxPPO


def build_aux_ppo(
    agent: TorchAgent,
    dataloader: BatchedDataLoader[SamplesForLoss[np.ndarray]],
    optimizer: Optimizer,
    learning_rate_scheduler: Optional[_LRScheduler],
    value_loss_coeff: float,
    entropy_loss_coeff: float,
    aux_loss_coeff: float,
    clip_grad_norm: Optional[float],
    epochs: int,
    ratio_clip: float,
    value_clipping_mode: str,
    value_clip: Optional[float] = None,
    kl_divergence_limit: float = np.inf,
):
    return AuxPPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        value_loss_coeff=value_loss_coeff,
        entropy_loss_coeff=entropy_loss_coeff,
        aux_loss_coeff=aux_loss_coeff,
        clip_grad_norm=clip_grad_norm,
        epochs=epochs,
        ratio_clip=ratio_clip,
        value_clipping_mode=value_clipping_mode,
        value_clip=value_clip,
        kl_divergence_limit=kl_divergence_limit,
    )


def build_ppo(
    agent: TorchAgent,
    dataloader: BatchedDataLoader[SamplesForLoss[np.ndarray]],
    optimizer: Optimizer,
    learning_rate_scheduler: Optional[_LRScheduler],
    value_loss_coeff: float,
    entropy_loss_coeff: float,
    clip_grad_norm: Optional[float],
    epochs: int,
    ratio_clip: float,
    value_clipping_mode: str,
    value_clip: Optional[float] = None,
    kl_divergence_limit: float = np.inf,
):
    # there is no logic here, but I want to keep all of hydra's instantiate targets in one place
    return PPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        value_loss_coeff=value_loss_coeff,
        entropy_loss_coeff=entropy_loss_coeff,
        clip_grad_norm=clip_grad_norm,
        epochs=epochs,
        ratio_clip=ratio_clip,
        value_clipping_mode=value_clipping_mode,
        value_clip=value_clip,
        kl_divergence_limit=kl_divergence_limit,
    )


def build_embedder(
    embedding_size: int,
    mlp_1_layers: list[int],
    mlp_2_layers: list[int],
    mlp_act: str,
    group_size: int,
    sampling_ratio: int,
    random_start: bool,
) -> Embedder:
    mlp_1 = MLP(mlp_1_layers, act=mlp_act)
    mlp_2_layers.append(embedding_size)
    mlp_2 = MLP(mlp_2_layers, act=mlp_act)
    return Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=group_size,
        sampling_ratio=sampling_ratio,
        random_start=random_start,
    )


def build_transformer_block(
    embedding_size: int,
    dropout: float,
    mlp_ratio: int,
    mlp_act: str,
    attention_num_heads: int,
    attention_bias: bool,
    attention_qkv_bias: bool,
) -> TransformerBlock:
    mlp_layers = [embedding_size, int(mlp_ratio * embedding_size), embedding_size]
    mlp = MLP(
        mlp_layers,
        act=mlp_act,
        norm=None,
        dropout=dropout,
    )
    attention = MultiheadAttention(
        embed_dim=embedding_size,
        num_heads=attention_num_heads,
        add_bias_kv=attention_qkv_bias,
        bias=attention_bias,
        dropout=dropout,
        batch_first=True,
    )
    return TransformerBlock(attention, mlp)


def build_transformer_encoder(
    transformer_block_factory: Callable[[], TransformerBlock], depth: int
):
    blocks = [transformer_block_factory() for _ in range(depth)]
    return TransformerEncoder(blocks)


def build_transformer_decoder(
    transformer_block_factory: Callable[[], TransformerBlock], depth: int
):
    blocks = [transformer_block_factory() for _ in range(depth)]
    return TransformerDecoder(blocks)


def build_pos_embedder(mlp_layers: list[int], act: type[nn.Module] | str) -> MLP:
    return MLP(mlp_layers, act=act, norm=None)


def build_masked_encoder(
    mask_ratio: float,
    mask_type: str,
    transformer_encoder: TransformerEncoder,
    pos_embedder: MLP,
):
    return MaskedEncoder(
        mask_ratio=mask_ratio,
        mask_type=mask_type,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )


def build_continuous_pg_model(
    embedder: Embedder,
    finetune_encoder: FinetuneEncoder,
    n_actions: int,
    mu_mlp_hidden_sizes: list[int],
    mu_mlp_act: type[nn.Module] | str,
    mu_out_act: Optional[type[nn.Module] | str],
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
    init_log_std: float,
):
    input_size = finetune_encoder.out_dim

    mu_hidden_nonlinearity = (
        getattr(torch.nn, mu_mlp_act) if isinstance(mu_mlp_act, str) else mu_mlp_act
    )

    mu_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=mu_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=mu_hidden_nonlinearity,
        output_size=n_actions,
    )
    if mu_out_act is not None:
        mu_out_nonlinearity = (
            getattr(torch.nn, mu_out_act) if isinstance(mu_out_act, str) else mu_out_act
        )
        mu_mlp = nn.Sequential(mu_mlp, mu_out_nonlinearity())

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )
    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )
    return ContinuousPgModel(
        embedder=embedder,
        encoder=finetune_encoder,
        mu_mlp=mu_mlp,
        value_mlp=value_mlp,
        init_log_std=init_log_std,
    )


def build_categorical_pg_model(
    embedder: Embedder,
    finetune_encoder: FinetuneEncoder,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
):
    input_size = finetune_encoder.out_dim

    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )

    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )

    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )

    return CategoricalPgModel(
        embedder=embedder,
        encoder=finetune_encoder,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
    )


def build_rl_aux_categorical_pg_model(
    embedder: Embedder,
    aux_mae: AuxMae,
    n_actions: int,
    pi_mlp_hidden_sizes: list[int],
    pi_mlp_act: type[nn.Module] | str,
    value_mlp_hidden_sizes: list[int],
    value_mlp_act: type[nn.Module] | str,
):
    input_size = aux_mae.out_dim
    pi_hidden_nonlinearity = (
        getattr(torch.nn, pi_mlp_act) if isinstance(pi_mlp_act, str) else pi_mlp_act
    )
    pi_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=pi_mlp_hidden_sizes,
        # hidden_nonlinearity=activation_resolver(pi_mlp_act),
        hidden_nonlinearity=pi_hidden_nonlinearity,
        output_size=n_actions,
    )

    value_hidden_nonlinearity = (
        getattr(torch.nn, value_mlp_act)
        if isinstance(value_mlp_act, str)
        else value_mlp_act
    )
    value_mlp = MlpModel(
        input_size=input_size,
        hidden_sizes=value_mlp_hidden_sizes,
        hidden_nonlinearity=value_hidden_nonlinearity,
        output_size=1,
    )

    return AuxMaeCategoricalPgModel(
        embedder=embedder,
        aux_mae=aux_mae,
        pi_mlp=pi_mlp,
        value_mlp=value_mlp,
    )


def build_finetune_encoder(
    transformer_encoder: TransformerEncoder,
    pos_embedder: nn.Module,
    mlp_head_hidden_sizes: list[int],
    mlp_head_out_size: int,
    mlp_head_act: type[nn.Module] | str,
):
    mlp_head_sizes = [2 * transformer_encoder.dim] + mlp_head_hidden_sizes
    mlp_head_sizes.append(mlp_head_out_size)
    mlp_head = MLP(channel_list=mlp_head_sizes, act=mlp_head_act, norm="layer_norm")

    return FinetuneEncoder(
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mlp_head=mlp_head,
    )


def build_aux_mae(
    masked_encoder: MaskedEncoder,
    masked_decoder: MaskedDecoder,
    mae_prediction_head: MaePredictionHead,
    mlp_head_hidden_sizes: list[int],
    mlp_head_out_size: int,
    mlp_head_act: type[nn.Module] | str,
):
    mlp_head_sizes = [2 * masked_encoder.dim] + mlp_head_hidden_sizes
    mlp_head_sizes.append(mlp_head_out_size)
    mlp_head = MLP(channel_list=mlp_head_sizes, act=mlp_head_act, norm="layer_norm")

    return AuxMae(
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=mae_prediction_head,
        mlp_head=mlp_head,
    )


def build_masked_autoencoder_modules(config):
    embedder_conf = config["model"]["embedder"]
    embedding_size = embedder_conf["embedding_size"]
    group_size = embedder_conf["group_size"]

    embedder = instantiate(config.model.embedder, _convert_="partial")
    model_conf = config["model"]
    blocks = []

    for _ in range(model_conf["encoder_depth"]):
        block = instantiate(
            config.model.transformer_block, embedding_size=embedding_size
        )
        blocks.append(block)

    transformer_encoder = TransformerEncoder(blocks)

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")

    masked_encoder = instantiate(
        config.model.masked_encoder,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    blocks = []
    for _ in range(model_conf["decoder_depth"]):
        block = instantiate(
            config.model.transformer_block, embedding_size=embedding_size
        )
        blocks.append(block)

    pos_embedder = instantiate(config.model.pos_embedder)
    transformer_decoder = TransformerDecoder(blocks)
    masked_decoder = MaskedDecoder(transformer_decoder, pos_embedder)
    prediction_head = MaePredictionHead(embedding_size, group_size)

    return embedder, masked_encoder, masked_decoder, prediction_head


def build_masked_autoencoder(config):
    (
        embedder,
        masked_encoder,
        masked_decoder,
        prediction_head,
    ) = build_masked_autoencoder_modules(config)
    masked_autoencoder = MaskedAutoEncoder(
        embedder=embedder,
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=prediction_head,
        learning_rate=config["learning_rate"],
    )

    return masked_autoencoder
