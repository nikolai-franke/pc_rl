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
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
                              add_reward_normalization)
from parllel.runners import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.agents.categorical import CategoricalPgAgent
from parllel.torch.algos.ppo import (PPO, BatchedDataLoader,
                                     build_dataloader_buffer)
from parllel.torch.distributions import Categorical
from parllel.torch.handler import TorchHandler
from parllel.torch.utils import buffer_to_device, torchify_buffer
from parllel.transforms import Compose
from parllel.types import BatchSpec
from torch.nn import ModuleList
from torch_geometric.nn import MLP

import wandb
from pc_rl.envs.reach_env import build_reach_env
from pc_rl.models.modules.transformer import (FinetuneEncoder,
                                              TransformerEncoder)
from pc_rl.models.pg_model import PgModel


@contextmanager
def build(config: DictConfig):
    # Parllel
    parallel = config["parallel"]
    storage = "shared" if parallel else "local"
    discount = config["algo"]["discount"]
    batch_spec = BatchSpec(config["batch_T"], config["batch_B"])
    TrajInfo.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage

    cage_kwargs = dict(
        EnvClass=build_reach_env,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
    )
    example_cage = CageCls(**cage_kwargs)
    example_cage.random_step_async()
    action, obs, reward, terminated, truncated, info = example_cage.await_step()

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
        shape=(8000, 3),
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
    group_size = embedder_conf.group_size

    embedder = instantiate(config.model.embedder, _convert_="partial")

    model_conf = config["model"]
    blocks = []
    for _ in range(model_conf["encoder_depth"]):
        block = instantiate(
            config.model.transformer_block, embedding_size=embedding_size
        )
        blocks.append(block)

    blocks = ModuleList(blocks)
    transformer_encoder = TransformerEncoder(blocks)

    pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    mlp_head = MLP(
        channel_list=[2 * embedding_size, 512, 1024], act="relu", norm="layer_norm"
    )
    encoder = FinetuneEncoder(
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
        mlp_head=mlp_head,
    )

    rl_model_conf = model_conf["rl_model"]

    pg_model = PgModel(
        embedder=embedder,
        encoder=encoder,
        n_actions=n_actions,
        mlp_hidden_sizes=list(rl_model_conf["mlp_layers"]),
        mlp_act=torch.nn.Tanh,
    )

    batch_env.observation[0] = obs_space.sample().pos
    example_obs = batch_env.observation[0]

    agent = CategoricalPgAgent(
        model=pg_model,
        distribution=distribution,
        example_obs=example_obs,
        device=device,
    )
    agent = TorchHandler(agent)

    _, agent_info = agent.step(example_obs)
    storage = "shared" if parallel else "local"
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

    algorithm = PPO(
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

    runner = OnPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        n_steps=runner_config["n_steps"],
        log_interval_steps=runner_config["log_interval_steps"],
    )
    try:
        yield runner

    finally:
        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        buffer_method(batch_buffer, "close")
        buffer_method(batch_buffer, "destroy")


@hydra.main(version_base=None, config_path="../conf", config_name="train_ppo")
def main(config: DictConfig):
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",
        project="pc_rl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,
        save_code=True,
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
