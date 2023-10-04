from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from torch.optim.adamw import AdamW
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.transforms import (Compose, FixedPoints, GridSampling,
                                        NormalizeScale, RandomRotate,
                                        SamplePoints)

import pc_rl.builder  # for hydra's instantiate
import wandb
from pc_rl.callbacks.log_pointclouds import LogPointCloudCallback
from pc_rl.datasets.in_memory import PcInMemoryDataset
from pc_rl.models.mae import MaskedAutoEncoder
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder
from pc_rl.utils.aux_loss import get_loss_fn


@hydra.main(version_base=None, config_path="../conf", config_name="mae_pretrain")
def main(config: DictConfig):
    embedder = instantiate(config.model.embedder, _convert_="partial")

    transformer_block_factory = instantiate(
        config.model.transformer_block,
        embedding_size=config.model.embedder.embedding_size,
        _partial_=True,
    )

    transformer_encoder = instantiate(
        config.model.transformer_encoder,
        transformer_block_factory=transformer_block_factory,
    )

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")

    masked_encoder = instantiate(
        config.model.masked_encoder,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    transformer_decoder = instantiate(
        config.model.transformer_decoder,
        transformer_block_factory=transformer_block_factory,
    )
    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    masked_decoder = MaskedDecoder(
        transformer_decoder=transformer_decoder, pos_embedder=pos_embedder
    )

    mae_prediction_head = MaePredictionHead(
        dim=config.model.embedder.embedding_size,
        group_size=config.model.embedder.group_size,
        n_out_channels=3 if not config.use_color else 6,
    )

    masked_autoencoder = MaskedAutoEncoder(
        embedder=embedder,
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=mae_prediction_head,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    transforms = []
    if (downsampling_method := config.dataset.downsampling_method) == "fixed_points":
        transforms.append(
            FixedPoints(
                config.dataset.num_points, replace=False, allow_duplicates=False
            )
        )
    elif downsampling_method == "voxel_grid":
        transforms.append(GridSampling(config.dataset.grid_size))

    transforms.append(RandomRotate(180, 0))
    transforms.append(RandomRotate(180, 1))
    transforms.append(RandomRotate(180, 2))
    transforms.append(NormalizeScale())
    transform = Compose(transforms)

    path = str(Path(__file__).parent.resolve() / config.dataset.path)
    # if dataset_name == "modelnet_10":
    #     dataset = ModelNet(path, "10", True, transform)
    #     validation_dataset = dataset[int(0.9 * len(dataset)) :]
    #     dataset = dataset[: int(0.9 * len(dataset))]
    # elif dataset_name == "modelnet_40":
    #     dataset = ModelNet(path, "40", True, transform)
    #     validation_dataset = dataset[int(0.9 * len(dataset)) :]
    #     dataset = dataset[: int(0.9 * len(dataset))]
    if config.dataset.name == "shapenet":
        dataset = ShapeNet(
            path,
            include_normals=False,
            transform=transform,
            split="train",
        )
        validation_dataset = ShapeNet(
            path,
            include_normals=False,
            transform=transform,
            split="val",
        )
    elif config.dataset.name in (
        "reach",
        "thread_in_hole",
        "color_thread_in_hole",
        "rope_cutting",
    ):
        dataset = PcInMemoryDataset(root=path, transform=transform)
        validation_dataset = dataset[int(0.9 * len(dataset)) :]
        dataset = dataset[: int(0.9 * len(dataset))]
    else:
        raise NotImplementedError

    data_module = LightningDataset(
        dataset,
        val_dataset=validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    wandb_logger = WandbLogger(project="MAE", log_model=True)
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    log_point_cloud_callback = LogPointCloudCallback()

    wandb_logger.experiment.config.update(wandb_config)

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["max_epochs"],
        log_every_n_steps=config["log_every_n_steps"],
        callbacks=[log_point_cloud_callback],
    )
    trainer.fit(
        masked_autoencoder,
        data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    wandb.finish()


if __name__ == "__main__":
    main()
