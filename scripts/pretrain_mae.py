from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import ModelNet, ShapeNet
from torch_geometric.transforms import (Compose, FixedPoints, NormalizeScale,
                                        SamplePoints)

from pc_rl.builder import build_masked_autoencoder
from pc_rl.callbacks.log_pointclouds import LogPointCloudCallback


@hydra.main(version_base=None, config_path="../conf", config_name="mae_pretrain")
def main(config: DictConfig):
    masked_autoencoder = build_masked_autoencoder(config)
    dataset_conf = config["dataset"]

    transform = Compose([SamplePoints(dataset_conf["num_points"]), NormalizeScale()])

    path = str(Path(__file__).parent.resolve() / dataset_conf["path"])
    if (dataset_name := dataset_conf["name"]) == "modelnet_10":
        dataset = ModelNet(path, "10", True, transform)
    elif dataset_name == "modelnet_40":
        dataset = ModelNet(path, "40", True, transform)
    elif dataset_name == "shapenet":
        transform = Compose(
            [
                FixedPoints(
                    dataset_conf["num_points"], replace=False, allow_duplicates=False
                ),
                NormalizeScale(),
            ]
        )
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
    else:
        raise NotImplementedError

    data_module = LightningDataset(
        dataset,
        val_dataset=validation_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    wandb_logger = WandbLogger(project="MAE")
    wandb_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    log_point_cloud_callback = LogPointCloudCallback()

    for k, v in wandb_config.items():  # type: ignore
        wandb_logger.experiment.config[k] = v

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


if __name__ == "__main__":
    main()
