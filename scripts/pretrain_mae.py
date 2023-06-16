from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import NormalizeScale, SamplePoints

from pc_rl.builder import build_masked_autoencoder


@hydra.main(version_base=None, config_path="../conf", config_name="mae_pretrain")
def main(config: DictConfig):
    masked_autoencoder = build_masked_autoencoder(config)
    transform, pre_transform = NormalizeScale(), SamplePoints(1024)

    path = str(Path(__file__).parent.resolve() / "../data/ModelNet40")
    dataset = ModelNet(path, "40", True, transform, pre_transform)

    data_module = LightningDataset(dataset, batch_size=128, num_workers=4)

    wandb_logger = WandbLogger(project="MAE")
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100, log_every_n_steps=5)
    trainer.fit(masked_autoencoder, data_module)


if __name__ == "__main__":
    main()
