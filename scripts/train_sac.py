import itertools
import multiprocessing as mp
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import parllel.logger as logger
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from parllel import Array, ArrayDict, dict_map
from parllel.cages import ProcessCage, SerialCage, TrajInfo
from parllel.logger import Verbosity
from parllel.patterns import build_cages_and_sample_tree, build_eval_sampler
from parllel.replays.replay import ReplayBuffer
from parllel.runners import OffPolicyRunner
from parllel.samplers import BasicSampler
from parllel.torch.algos.sac import build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.types import BatchSpec

import pc_rl.builder  # for hydra's instantiate
import wandb
from pc_rl.agents.sac import SacAgent
from pc_rl.algos.sac import PcSac
from pc_rl.models.finetune_encoder import FinetuneEncoder


@contextmanager
def build(config: DictConfig):
    parallel = config.parallel
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
        full_size=config.algo.replay_length,
        keys_to_skip="observation",
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    n_actions = action_space.shape[0]

    distribution = SquashedGaussian(
        dim=n_actions,
        scale=action_space.high[0],
    )

    sample_tree["observation"] = dict_map(
        Array.from_numpy,
        metadata.example_obs,
        feature_shape=obs_space.shape,
        batch_shape=tuple(batch_spec),
        kind="jagged",
        storage="managed" if parallel else "local",
        padding=1,
        full_size=config.algo.replay_length,
    )
    sample_tree["observation"][0] = obs_space.sample()
    example_obs_batch = sample_tree["observation"][0]

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

    mlp_input_size = finetune_encoder.out_dim

    pi_model = instantiate(
        config.model.pi_mlp_head,
        input_size=mlp_input_size,
        action_size=n_actions,
        _convert_="partial",
    )
    q1_model = instantiate(
        config.model.q_mlp_head,
        input_size=mlp_input_size,
        action_size=n_actions,
        _convert_="partial",
    )
    q2_model = instantiate(
        config.model.q_mlp_head,
        input_size=mlp_input_size,
        action_size=n_actions,
        _convert_="partial",
    )

    model = torch.nn.ModuleDict(
        {
            "pi": pi_model,
            "q1": q1_model,
            "q2": q2_model,
            "embedder": embedder,
            "encoder": finetune_encoder,
        }
    )

    # instantiate agent
    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config.algo.learning_starts,
    )

    sampler = BasicSampler(
        batch_spec=batch_spec,
        envs=cages,
        agent=agent,
        sample_tree=sample_tree,
        max_steps_decorrelate=config.max_steps_decorrelate,
        get_bootstrap_value=False,
    )

    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=config["algo"]["replay_length"],
        replay_batch_size=config["algo"]["batch_size"],
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=batch_transform,
    )

    optimizers = {
        "pi": torch.optim.Adam(
            itertools.chain(
                agent.model["embedder"].parameters(),
                agent.model["encoder"].parameters(),
                agent.model["pi"].parameters(),
            ),
            lr=config.algo.learning_rate,
            **config.get("optimizer", {}),
        ),
        "q": torch.optim.Adam(
            itertools.chain(
                agent.model["q1"].parameters(),
                agent.model["q2"].parameters(),
            ),
            lr=config.algo.learning_rate,
            **config.get("optimizer", {}),
        ),
    }

    # create algorithm
    algorithm = PcSac(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        optimizers=optimizers,
        **config.algo,
    )

    eval_sampler, eval_sample_tree = build_eval_sampler(
        sample_tree=sample_tree,
        agent=agent,
        CageCls=CageCls,
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": False},
        TrajInfoClass=TrajInfo,
        **config.eval,
    )

    eval_sample_tree["observation"][0] = obs_space.sample()

    # create runner
    runner = OffPolicyRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        **config["runner"],
    )

    try:
        yield runner

    finally:
        eval_cages = eval_sampler.envs
        eval_sampler.close()
        for cage in eval_cages:
            cage.close()
        eval_sample_tree.close()

        sampler.close()
        agent.close()
        for cage in cages:
            cage.close()
        sample_tree.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    mp.set_start_method("fork")

    run = wandb.init(
        anonymous="must",  # for this example, send to wandb dummy account
        project="pc_rl",
        tags=["continuous", "state-based", "sac", "feedforward"],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        # mode="disabled",
    )

    logger.init(
        wandb_run=run,
        # this log_dir is used if wandb is disabled (using `wandb disabled`)
        log_dir=Path(f"log_data/pc_rl/sac/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"),
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
