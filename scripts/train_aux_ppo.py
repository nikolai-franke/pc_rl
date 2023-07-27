import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import parllel.logger as logger
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from parllel import Array, ArrayDict, dict_map
from parllel.cages import ProcessCage, SerialCage, TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_agent_info,
                              add_bootstrap_value, add_reward_normalization,
                              build_cages_and_sample_tree, build_eval_sampler)
from parllel.runners import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.algos.ppo import BatchedDataLoader, build_loss_sample_tree
from parllel.torch.distributions import Categorical, Gaussian
from parllel.transforms import Compose
from parllel.transforms.video_recorder import RecordVectorizedVideo
from parllel.types import BatchSpec

import pc_rl.builder  # import for hydra's instantiate
import wandb
from pc_rl.agents.aux_categorical import MaeCategoricalPgAgent
from pc_rl.algos.aux_ppo import AuxPPO
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder


@contextmanager
def build(config: DictConfig):
    # Parllel
    parallel = config.parallel
    storage = "shared" if parallel else "local"
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    TrajInfo.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage

    env_factory = instantiate(config["env"], _convert_="object", _partial_=True)

    cages, sample_tree, metadata = build_cages_and_sample_tree(
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": False},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
        keys_to_skip="observation",
    )

    obs_space, action_space = metadata.obs_space, metadata.action_space
    discrete = isinstance(action_space, gym.spaces.Discrete)
    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        metadata.example_obs,
        feature_shape=obs_space.shape,
        batch_shape=tuple(batch_spec),
        kind="jagged",
        storage="managed" if parallel else "local",
        padding=1,
    )

    sample_tree["observation"][0] = obs_space.sample()
    example_obs_batch = sample_tree["observation"][0]

    n_actions = action_space.n if discrete else action_space.shape[0]
    distribution = Gaussian(dim=n_actions)

    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    sample_tree["observation"][0] = obs_space.sample()
    example_obs_batch = sample_tree["observation"][0]

    transformer_block_factory = instantiate(
        config.model.transformer_block,
        embedding_size=config.model.embedder.embedding_size,
        _partial_=True,
    )
    transformer_encoder = instantiate(
        config.model.masked_encoder.transformer_encoder,
        transformer_block_factory=transformer_block_factory,
    )

    transformer_decoder = instantiate(
        config.model.masked_decoder.transformer_decoder,
        transformer_block_factory=transformer_block_factory,
    )

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    embedder = instantiate(config.model.embedder, _convert_="partial")

    masked_encoder = instantiate(
        config.model.masked_encoder,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    masked_decoder = MaskedDecoder(
        transformer_decoder=transformer_decoder,
        pos_embedder=pos_embedder,
    )
    mae_prediction_head = MaePredictionHead(
        dim=config.model.embedder.embedding_size,
        group_size=config.model.embedder.group_size,
    )

    aux_mae = instantiate(
        config.model.aux_mae,
        _convert_="partial",
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=mae_prediction_head,
    )

    aux_mae_pg_model = instantiate(
        config.model.rl_model,
        _convert_="partial",
        embedder=embedder,
        aux_mae=aux_mae,
        n_actions=n_actions,
    )

    agent = GaussianPgAgent(
        model=aux_mae_pg_model,
        distribution=distribution,
        example_obs=example_obs_batch,
        device=device,
    )

    sample_tree = add_agent_info(sample_tree, agent, example_obs_batch)
    sample_tree = add_bootstrap_value(sample_tree)

    batch_transforms, step_transforms = [], []

    sample_tree, batch_transforms = add_reward_normalization(
        sample_tree,
        batch_transforms,
        discount=discount,
    )

    sample_tree, batch_transforms = add_advantage_estimation(
        sample_tree,
        batch_transforms,
        discount=discount,
        gae_lambda=config.algo.gae_lambda,
        normalize=config.algo.normalize_advantage,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config.max_steps_decorrelate,
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    loss_sample_tree = build_loss_sample_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    dataloader = BatchedDataLoader(
        tree=loss_sample_tree,
        sampler_batch_spec=batch_spec,
        n_batches=config.algo.minibatches,
        batch_transform=batch_transform,
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=config.algo.learning_rate,
        **config.get("optimizer", {}),
    )

    algorithm = instantiate(
        config.algo,
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        _convert_="partial",
    )

    eval_sampler, eval_sample_tree = build_eval_sampler(
        sample_tree=sample_tree,
        agent=agent,
        CageCls=CageCls,
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": True},
        TrajInfoClass=TrajInfo,
        n_eval_envs=config.eval.n_eval_envs,
        max_traj_length=config.eval.max_traj_length,
        min_trajectories=config.eval.min_trajectories,
        step_transforms=step_transforms,
    )

    eval_sample_tree["observation"][0] = obs_space.sample()


    # video_recorder = RecordVectorizedVideo(
    #     batch_buffer=step_buffer,
    #     buffer_key_to_record="env_info.rendering",
    #     env_fps=50,
    #     record_every_n_steps=1,
    #     output_dir=Path(f"videos/pc_rl/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
    #     video_length=config.env.max_episode_steps,
    # )

    # step_transforms.append(video_recorder)

    # if step_transforms is not None:
    #     step_transforms = Compose(step_transforms)


    runner = OnPolicyRunner(
        sampler=sampler,
        eval_sampler=eval_sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
    )
    try:
        yield runner

    finally:
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_aux_ppo")
def main(config: DictConfig):
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",
        project="pc_rl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,
        save_code=True,
        # mode="disabled",
    )
    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/pc_rl/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        tensorboard=True,
        output_files={
            "txt": "log.txt",
            # "csv": "progress.csv",
        },
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        model_save_path="model.pt",
        verbosity=Verbosity.DEBUG,
    )
    with build(config) as runner:
        runner.run()
    run.finish()


if __name__ == "__main__":
    main()
