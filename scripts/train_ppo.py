import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import parllel.logger as logger
import torch
from gymnasium.spaces import Discrete
from omegaconf import DictConfig, OmegaConf
from parllel.arrays import buffer_from_example
from parllel.buffers import AgentSamples, Samples, buffer_method
from parllel.cages import TrajInfo
from parllel.patterns import (add_advantage_estimation, add_bootstrap_value,
                              add_obs_normalization, add_reward_normalization,
                              build_cages_and_env_buffers)
from parllel.runners import OnPolicyRunner
from parllel.samplers.basic import BasicSampler
from parllel.torch.agents.gaussian import Gaussian, GaussianPgAgent
from parllel.torch.algos.ppo import (PPO, BatchedDataLoader,
                                     build_dataloader_buffer)
from parllel.torch.handler import TorchHandler
from parllel.transforms import Compose
from parllel.types import BatchSpec
from torch.nn import ModuleList, MultiheadAttention
from torch_geometric.nn import MLP

import wandb
from pc_rl.envs.reach_env import build_reach_env
from pc_rl.models.modules.embedder import Embedder
from pc_rl.models.modules.transformer import (MaskedEncoder, TransformerBlock,
                                              TransformerEncoder)
from pc_rl.models.pg_model import PgModel


@contextmanager
def build(config: DictConfig):
    # Parllel
    batch_spec = BatchSpec(128, 2)
    TrajInfo.set_discount(0.99)
    parallel = False

    cages, batch_action, batch_env = build_cages_and_env_buffers(
        EnvClass=build_reach_env,
        env_kwargs={},
        TrajInfoClass=TrajInfo,
        reset_automatically=True,
        batch_spec=batch_spec,
        parallel=parallel,
    )

    spaces = cages[0].spaces
    obs_space, action_space = spaces.observation, spaces.action
    n_actions = action_space.n
    distribution = Gaussian(dim=n_actions)  # type: ignore
    device = torch.device("cpu")

    embedder_conf = config["embedder"]
    embedding_size = embedder_conf["embedding_size"]
    group_size = embedder_conf["group_size"]

    mlp_1 = MLP(list(embedder_conf["mlp_1_layers"]), act=embedder_conf["act"])
    mlp_2_layers = list(embedder_conf["mlp_2_layers"])
    mlp_2_layers.append(embedding_size)
    mlp_2 = MLP(mlp_2_layers, act=embedder_conf["act"])
    embedder = Embedder(
        mlp_1=mlp_1,
        mlp_2=mlp_2,
        group_size=group_size,
        sampling_ratio=embedder_conf["sampling_ratio"],
        random_start=embedder_conf["random_start"],
    )

    model_conf = config["model"]
    attention_conf = model_conf["attention"]
    block_conf = model_conf["transformer_block"]
    blocks = []
    for _ in range(model_conf["encoder_depth"]):
        mlp = MLP(
            list(block_conf["mlp_layers"]),
            act=block_conf["act"],
            norm=None,
            dropout=block_conf["dropout"],
        )
        attention = MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=attention_conf["num_heads"],
            add_bias_kv=attention_conf["qkv_bias"],
            dropout=attention_conf["dropout"],
            bias=attention_conf["bias"],
            batch_first=True,
        )
        blocks.append(TransformerBlock(attention, mlp))

    blocks = ModuleList(blocks)

    transformer_encoder = TransformerEncoder(blocks)

    pos_embedder = MLP(
        list(model_conf["pos_embedder"]["mlp_layers"]),
        act=model_conf["pos_embedder"]["act"],
        norm=None,
    )
    masked_encoder = MaskedEncoder(
        mask_ratio=model_conf["mask_ratio"],
        transformer_encoder=transformer_encoder,
        pos_embedder=pos_embedder,
    )

    rl_model_conf = model_conf["rl_model"]

    pg_model = PgModel(
        embedder=embedder,
        encoder=masked_encoder,
        decoder=None,
        n_actions=n_actions,
        mlp_hidden_sizes=list(rl_model_conf["mlp_layers"]),
        mlp_act=torch.nn.Tanh,
    )

    agent = GaussianPgAgent(
        model=pg_model,
        distribution=distribution,
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )
    agent = TorchHandler(agent)

    batch_env.observation[0] = obs_space.sample()
    example_obs = batch_env.observation[0]

    _, agent_info = agent.step(example_obs)
    storage = "shared" if parallel else "local"
    batch_agent_info = buffer_from_example(agent_info, (batch_spec.T,), storage=storage)
    batch_agent = AgentSamples(batch_action, batch_agent_info)
    batch_buffer = Samples(batch_agent, batch_env)

    batch_buffer = add_bootstrap_value(batch_buffer)
    batch_transforms, step_transforms = [], []

    batch_buffer, step_transforms = add_obs_normalization(
        batch_buffer,
        step_transforms,
        initial_count=10_000,
    )

    batch_buffer, batch_transforms = add_reward_normalization(
        batch_buffer,
        batch_transforms,
        discount=0.99,
    )

    batch_buffer, batch_transforms = add_advantage_estimation(
        batch_buffer,
        batch_transforms,
        discount=0.99,
        gae_lambda=0.95,
        normalize=True,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_buffer=batch_buffer,
        max_steps_decorrelate=20,
        get_bootstrap_value=True,
        obs_transform=Compose(step_transforms),
        batch_transform=Compose(batch_transforms),
    )

    dataloader_buffer = build_dataloader_buffer(batch_buffer)

    dataloader = BatchedDataLoader(
        buffer=dataloader_buffer,
        sampler_batch_spec=batch_spec,
        n_batches=4,
    )

    optimizer = torch.optim.Adam(
        agent.model.parameters(),
        lr=1e-5,
    )

    algorithm = PPO(agent=agent, dataloader=dataloader, optimizer=optimizer)

    runner = OnPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        n_steps=102_400,
        log_interval_steps=10_240,
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


@hydra.main(version_base=None, config_path="../conf", config_name="ppo")
def main(config: DictConfig):
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",
        mode="disabled",
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
    run.finis()


if __name__ == "__main__":
    main()
