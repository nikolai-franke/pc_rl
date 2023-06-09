import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import parllel.logger as logger
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from parllel.arrays import Array, buffer_from_dict_example, buffer_from_example
from parllel.buffers import (AgentSamples, Buffer, EnvSamples, Samples,
                             buffer_asarray, buffer_method)
from parllel.cages import ProcessCage, SerialCage, TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import (EvalSampler, add_advantage_estimation,
                              add_bootstrap_value, add_reward_normalization,)
from parllel.runners import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.algos.ppo import (BatchedDataLoader,
                                     build_dataloader_buffer)
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.torch.models import MlpModel
from parllel.torch.utils import buffer_to_device, torchify_buffer
from parllel.transforms import Compose
from parllel.transforms.video_recorder import RecordVectorizedVideo
from parllel.types import BatchSpec
from torch_geometric.nn.models.mlp import MLP
from pc_rl.algos.aux_ppo import AuxPPO

import pc_rl.builder  # import for hydra's instantiate
import wandb
from pc_rl.agents.aux_categorical import MaeCategoricalPgAgent
from pc_rl.models.aux_mae_discrete_pg import AuxMaeDiscretePgModel
from pc_rl.models.modules.auxiliarey_mae import AuxiliaryMae
from pc_rl.models.modules.mae import (MaskedDecoder, MaskedEncoder,
                                      PredictionHead)


@contextmanager
def build(config: DictConfig):
    # Parllel
    parallel = config["parallel"]
    storage = "shared" if parallel else "local"
    discount = config["algo"]["discount"]
    batch_spec = BatchSpec(config["batch_T"], config["batch_B"])
    TrajInfo.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage

    EnvClass = instantiate(config["env"], _convert_="object", _partial_=True)

    cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
    )
    example_cage = CageCls(**cage_kwargs)
    example_cage.random_step_async()
    action, obs, reward, terminated, truncated, info = example_cage.await_step()  # type: ignore

    spaces = example_cage.spaces
    obs_space, action_space = spaces.observation, spaces.action

    example_cage.close()

    # allocate batch buffer based on examples
    np_obs = np.asanyarray(obs)
    if (dtype := np_obs.dtype) == np.float64:
        dtype = np.float32
    elif dtype == np.int64:
        dtype = np.int32

    batch_observation = Array(
        shape=(128 * 128, 3),
        dtype=dtype,
        batch_shape=tuple(batch_spec),
        kind="jagged",
        storage=storage,
        padding=1,
    )
    # in case environment creates rewards of shape (1,) or of integer type,
    # force to be correct shape and type
    batch_reward = buffer_from_dict_example(
        reward,
        tuple(batch_spec),
        name="reward",
        shape=(),
        dtype=np.float32,
        storage=storage,
    )
    batch_terminated = buffer_from_example(
        terminated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
    )
    batch_truncated = buffer_from_example(
        truncated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
    )
    batch_done = buffer_from_example(
        truncated,
        tuple(batch_spec),
        shape=(),
        dtype=bool,
        storage=storage,
        padding=1,
    )
    batch_info = buffer_from_dict_example(
        info, tuple(batch_spec), name="envinfo", storage=storage, padding=1
    )
    batch_env = EnvSamples(
        batch_observation,
        batch_reward,
        batch_done,
        batch_terminated,
        batch_truncated,
        batch_info,
    )

    batch_action = buffer_from_dict_example(
        action,
        tuple(batch_spec),
        name="action",
        force_32bit="float",
        storage=storage,
    )
    if storage == "shared":
        cage_kwargs["buffers"] = (
            batch_action,
            batch_observation,
            batch_reward,
            batch_done,
            batch_terminated,
            batch_truncated,
            batch_info,
        )

    logger.debug(f"Instantiating {batch_spec.B} environments...")
    cages = [CageCls(**cage_kwargs) for _ in range(batch_spec.B)]
    logger.debug("Environments instantiated.")

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action

    n_actions = action_space.n
    distribution = Categorical(dim=n_actions)  # type: ignore

    device = config["device"] or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    embedder_conf = config.model.embedder
    embedding_size = embedder_conf.embedding_size

    transformer_block_factory = instantiate(
        config.model.transformer_block, embedding_size=embedding_size, _partial_=True
    )
    transformer_encoder = instantiate(
        config.model.transformer_encoder,
        transformer_block_factory=transformer_block_factory,
    )

    transformer_decoder = instantiate(
        config.model.transformer_decoder,
        transformer_block_factory=transformer_block_factory,
    )

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    embedder = instantiate(config.model.embedder, _convert_="partial")

    masked_encoder = MaskedEncoder(
        mask_ratio=0.6,
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    masked_decoder = MaskedDecoder(
        transformer_decoder=transformer_decoder,
        pos_embedder=pos_embedder,
    )
    mae_prediction_head = PredictionHead(dim=512, group_size=32)

    mlp_head_sizes = [1024, 512, 1024]
    mlp_head = MLP(channel_list=mlp_head_sizes, act="relu", norm="layer_norm")
    aux_mae = AuxiliaryMae(
        masked_encoder, masked_decoder, mae_prediction_head, mlp_head
    )
    pi_mlp = MlpModel(
        input_size=1024,
        hidden_sizes=[256, 128],
        hidden_nonlinearity=torch.nn.Tanh,
        output_size=7,
    )
    value_mlp = MlpModel(
        input_size=1024,
        hidden_sizes=[256, 128],
        hidden_nonlinearity=torch.nn.Tanh,
        output_size=1,
    )

    aux_mae_pg_model = AuxMaeDiscretePgModel(
        embedder=embedder, auxiliary_mae=aux_mae, pi_mlp=pi_mlp, value_mlp=value_mlp
    )

    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]

    agent = MaeCategoricalPgAgent(
        model=aux_mae_pg_model,
        distribution=distribution,
        example_obs=example_obs,
        device=device,
    )

    agent = TorchHandler(agent)

    _, agent_info = agent.step(example_obs)
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), storage=storage)
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    batch_buffer = add_bootstrap_value(batch_buffer)
    batch_transforms, step_transforms = [], []

    batch_buffer, batch_transforms = add_reward_normalization(
        batch_buffer,
        batch_transforms,
        discount=discount,
    )
    algo_config = config["algo"]

    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=discount,
        gae_lambda=algo_config["gae_lambda"],
        normalize=algo_config["normalize_advantage"],
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=config["max_steps_decorrelate"],
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_dataloader_buffer(batch_buffer)

    def batch_transform(x: Buffer):
        x = buffer_asarray(x)
        x = torchify_buffer(x)
        return buffer_to_device(x, device=device)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
        sampler_batch_spec=batch_spec,
        batch_transform=batch_transform,
        n_batches=algo_config["minibatches"],
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=algo_config["learning_rate"],
    )

    algorithm = AuxPPO(
        agent=agent,
        dataloader=dataloader,
        optimizer=optimizer,
        learning_rate_scheduler=algo_config["learning_rate_scheduler"],
        value_loss_coeff=algo_config["value_loss_coeff"],
        entropy_loss_coeff=algo_config["entropy_loss_coeff"],
        clip_grad_norm=algo_config["clip_grad_norm"],
        epochs=algo_config["epochs"],
        ratio_clip=algo_config["ratio_clip"],
        value_clipping_mode=algo_config["value_clipping_mode"],
    )
    runner_config = config["runner"]

    eval_config = config["eval"]

    eval_cage_kwargs = dict(
        EnvClass=EnvClass,
        env_kwargs={"add_obs_to_info_dict": True},
        # env_kwargs={},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
    )

    example_cage = CageCls(**eval_cage_kwargs)
    example_cage.random_step_async()
    action, obs, reward, terminated, truncated, info = example_cage.await_step()  # type: ignore
    example_cage.close()

    step_shape = (1, eval_config["n_eval_envs"])

    step_observation = Array(
        shape=(128 * 128, 3),
        dtype=dtype,
        batch_shape=step_shape,
        kind="jagged",
        storage=storage,
        padding=1,
    )
    # in case environment creates rewards of shape (1,) or of integer type,
    # force to be correct shape and type
    step_reward = buffer_from_dict_example(
        reward,
        step_shape,
        name="reward",
        shape=(),
        dtype=np.float32,
        storage=storage,
    )
    step_terminated = buffer_from_example(
        terminated,
        step_shape,
        shape=(),
        dtype=bool,
        storage=storage,
    )
    step_truncated = buffer_from_example(
        truncated,
        step_shape,
        shape=(),
        dtype=bool,
        storage=storage,
    )
    step_done = buffer_from_example(
        truncated,
        step_shape,
        shape=(),
        dtype=bool,
        storage=storage,
        padding=1,
    )
    step_info = buffer_from_dict_example(
        info, step_shape, name="envinfo", storage=storage, padding=1
    )

    step_env = EnvSamples(
        observation=step_observation,
        reward=step_reward,
        done=step_done,
        terminated=step_terminated,
        truncated=step_truncated,
        env_info=step_info,
    )

    step_action = buffer_from_dict_example(
        action,
        step_shape,
        name="action",
        force_32bit="float",
        storage=storage,
    )

    step_env.observation[0] = obs_space.sample()
    example_obs = step_env.observation[0]
    _, agent_info = agent.step(example_obs)

    step_agent_info = buffer_from_example(agent_info, (1,), storage=storage)
    step_agent = AgentSamples(step_action, step_agent_info)
    step_buffer = Samples(agent=step_agent, env=step_env)

    if issubclass(CageCls, ProcessCage):
        eval_cage_kwargs["buffers"] = step_buffer

    eval_envs = [CageCls(**eval_cage_kwargs) for _ in range(eval_config["n_eval_envs"])]

    video_recorder = RecordVectorizedVideo(
        batch_buffer=step_buffer,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        record_every_n_steps=1,
        output_dir=Path(f"videos/pc_rl/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
        video_length=100,
    )

    step_transforms.append(video_recorder)

    if step_transforms is not None:
        step_transforms = Compose(step_transforms)

    eval_sampler = EvalSampler(
        max_traj_length=eval_config["max_traj_length"],
        min_trajectories=eval_config["min_trajectories"],
        envs=eval_envs,
        agent=agent,
        step_buffer=step_buffer,
        obs_transform=step_transforms,
        deterministic_actions=eval_config["deterministic_actions"],
    )
    # NOTE: end copy

    runner = OnPolicyRunner(
        sampler=sampler,
        eval_sampler=eval_sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        n_steps=runner_config["n_steps"],
        log_interval_steps=runner_config["log_interval_steps"],
        eval_interval_steps=runner_config["eval_interval_steps"],
    )
    try:
        yield runner

    finally:
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_buffer, "close")
        # buffer_method(step_buffer, "close")


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
