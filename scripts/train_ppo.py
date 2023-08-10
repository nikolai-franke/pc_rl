import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import hydra
import parllel.logger as logger
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from parllel import Array, ArrayDict, dict_map
from parllel.cages import ProcessCage, SerialCage, TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (add_advantage_estimation, add_agent_info,
                              add_bootstrap_value, add_reward_normalization,
                              build_cages_and_sample_tree)
from parllel.runners import RLRunner
from parllel.samplers.basic import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.agents.gaussian import GaussianPgAgent
from parllel.torch.algos.ppo import BatchedDataLoader, build_loss_sample_tree
from parllel.torch.distributions import Categorical, Gaussian
from parllel.transforms import Compose
from parllel.transforms.video_recorder import RecordVectorizedVideo
from parllel.types import BatchSpec

import pc_rl.builder  # import for hydra's instantiate
import wandb
from pc_rl.models.finetune_encoder import FinetuneEncoder


@contextmanager
def build(config: DictConfig):
    # Parllel
    parallel = config.parallel
    storage = "shared" if parallel else "local"
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    TrajInfo.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage

    env_factory = instantiate(config.env, _convert_="object", _partial_=True)

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
    AgentClass = CategoricalPgAgent if discrete else GaussianPgAgent
    DistributionClass = Categorical if discrete else Gaussian

    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        metadata.example_obs,
        feature_shape=obs_space.shape[1:],
        max_mean_num_elem=obs_space.shape[0],
        batch_shape=tuple(batch_spec),
        kind="jagged",
        storage=storage,
        padding=1,
    )

    sample_tree["observation"][0] = obs_space.sample()
    example_obs_batch = sample_tree["observation"][0]

    n_actions = action_space.n if discrete else action_space.shape[0]
    distribution = DistributionClass(dim=n_actions)

    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

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
    embedder = instantiate(config.model.embedder, _convert_="partial")

    finetune_encoder = FinetuneEncoder(
        pos_embedder=pos_embedder, transformer_encoder=transformer_encoder
    )

    pg_model = instantiate(
        config.model.rl_model,
        _convert_="partial",
        embedder=embedder,
        finetune_encoder=finetune_encoder,
        n_actions=n_actions,
    )

    agent = AgentClass(
        model=pg_model,
        distribution=distribution,  # type: ignore
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

    optimizer_conf = config.get("optimizer", {})
    optimizer_conf = OmegaConf.to_container(
        optimizer_conf, resolve=True, throw_on_missing=True
    )
    per_module_conf = optimizer_conf.pop("per_module", {})  # type: ignore

    per_parameter_options = [
        {
            "params": agent.model.embedder.parameters(),
            **per_module_conf.get("embedder", {}),
        },
        {
            "params": agent.model.encoder.parameters(),
            **per_module_conf.get("encoder", {}),
        },
        {
            "params": agent.model.pi_mlp.parameters(),
            **per_module_conf.get("pi", {}),
        },
        {
            "params": agent.model.value_mlp.parameters(),
            **per_module_conf.get("value", {}),
        },
    ]
    if not discrete:
        per_parameter_options.append(
            {"params": agent.model.log_std, **per_module_conf.get("log_std", {})}
        )

    optimizer = torch.optim.Adam(per_parameter_options, **optimizer_conf)

    algorithm = instantiate(
        config.algo,
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        _convert_="partial",
    )

    eval_cage_kwargs = dict(
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": True},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
    )

    eval_cages = [CageCls(**eval_cage_kwargs) for _ in range(config.eval.n_eval_envs)]
    example_cage = CageCls(**eval_cage_kwargs)
    example_cage.random_step_async()
    *_, info = example_cage.await_step()
    example_cage.close()

    eval_tree_keys = [
        "action",
        "agent_info",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "done",
        "env_info",
    ]
    eval_tree_example = ArrayDict(
        {key: sample_tree[key] for key in eval_tree_keys if key != "env_info"}
    )
    eval_tree_example["env_info"] = dict_map(
        Array.from_numpy,
        info,
        batch_shape=tuple(batch_spec),
        storage=storage,
    )

    eval_sample_tree = eval_tree_example.new_array(
        batch_shape=(1, config.eval.n_eval_envs)
    )
    eval_sample_tree["observation"][0] = obs_space.sample()

    video_recorder = RecordVectorizedVideo(
        batch_buffer=eval_sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        record_every_n_steps=1,
        output_dir=Path(f"videos/pc_rl/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        video_length=config.env.max_episode_steps,
    )

    eval_sampler = EvalSampler(
        max_traj_length=config.eval.max_traj_length,
        max_trajectories=config.eval.max_trajectories,
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        obs_transform=video_recorder,
    )

    runner = RLRunner(
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


@hydra.main(version_base=None, config_path="../conf", config_name="train_ppo")
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
        # verbosity=Verbosity.DEBUG,
    )
    with build(config) as runner:
        runner.run()
    run.finish()


if __name__ == "__main__":
    main()
