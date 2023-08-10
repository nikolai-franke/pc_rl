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
from parllel.patterns import build_cages_and_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.algos.sac import SAC, build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms.video_recorder import RecordVectorizedVideo
from parllel.types import BatchSpec

import pc_rl.builder  # for hydra's instantiate
import wandb
from pc_rl.agents.aux_sac import AuxPcSacAgent
from pc_rl.models.aux_mae import AuxMae
from pc_rl.models.modules.mae_prediction_head import MaePredictionHead
from pc_rl.models.modules.masked_decoder import MaskedDecoder


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
        batch_shape=tuple(batch_spec),
        max_mean_num_elem=obs_space.shape[0],
        feature_shape=obs_space.shape[1:],
        kind="jagged",
        storage="shared" if parallel else "local",
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

    aux_mae = AuxMae(
        masked_encoder=masked_encoder,
        masked_decoder=masked_decoder,
        mae_prediction_head=mae_prediction_head,
    )
    mlp_input_size = aux_mae.out_dim

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
            "encoder": aux_mae,
        }
    )

    # instantiate agent
    agent = AuxPcSacAgent(
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

    optimizer_conf = config.get("optimizer", {})
    optimizer_conf = OmegaConf.to_container(
        optimizer_conf, resolve=True, throw_on_missing=True
    )
    per_module_conf = optimizer_conf.pop("per_module", {})  # type: ignore
    optimizers = {
        "pi": torch.optim.Adam(
            [
                {
                    "params": agent.model["embedder"].parameters(),
                    **per_module_conf.get("embedder", {}),
                },
                {
                    "params": agent.model["aux_mae"].parameters(),
                    **per_module_conf.get("encoder", {}),
                },
                {
                    "params": agent.model["pi"].parameters(),
                    **per_module_conf.get("pi", {}),
                },
            ],
            **optimizer_conf,
        ),
        "q": torch.optim.Adam(
            [
                {
                    "params": agent.model["q1"].parameters(),
                    **config.optimizer.get("q", {}),
                },
                {
                    "params": agent.model["q2"].parameters(),
                    **config.optimizer.get("q", {}),
                },
            ],
            **optimizer_conf,
        ),
    }

    # create algorithm
    algorithm = SAC(
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        optimizers=optimizers,
        **config.algo,
    )

    eval_cage_kwargs = dict(
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": False},
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
        storage="shared" if parallel else "local",
        feature_shape=obs_space.shape[1:],
        full_size=config.algo.replay_length,
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

    # create runner
    runner = RLRunner(
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
        log_dir=Path(
            f"log_data/pc_rl/sac/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        ),
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