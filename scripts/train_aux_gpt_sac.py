import multiprocessing as mp
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
import parllel.logger as logger
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from parllel import Array, ArrayDict, dict_map
from parllel.cages import ProcessCage, SerialCage
from parllel.callbacks.recording_schedule import RecordingSchedule
from parllel.logger import Verbosity
from parllel.patterns import build_cages, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler
from parllel.torch.algos.sac import build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms.vectorized_video import RecordVectorizedVideo
from parllel.types import BatchSpec

import pc_rl.builder
import pc_rl.models.sac.q_and_pi_heads
import wandb
from pc_rl.agents.aux_sac import AuxPcSacAgent
from pc_rl.models.aux_gpt import AuxGPT
from pc_rl.models.finetune_encoder import FinetuneEncoder
from pc_rl.models.modules.gpt_decoder import \
    GPTDecoder  # for hydra's instantiate
from pc_rl.models.modules.prediction_head import PredictionHead
from pc_rl.utils.mani_skill_traj_info import ManiTrajInfo
from pc_rl.utils.sofa_traj_info import SofaTrajInfo


@contextmanager
def build(config: DictConfig):
    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    if "env_id" in config["env"].keys():
        TrajInfoClass = ManiTrajInfo
    else:
        TrajInfoClass = SofaTrajInfo
    TrajInfoClass.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage
    storage = "shared" if parallel else "local"

    env_factory = instantiate(config.env, _convert_="object", _partial_=True)

    cages, metadata = build_cages(
        EnvClass=env_factory,
        n_envs=batch_spec.B,
        env_kwargs={"add_obs_to_info_dict": False},
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )
    replay_length = int(config.algo.replay_length) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
        keys_to_skip=("obs", "next_obs"),
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
        kind="jagged",
        storage=storage,
        padding=1,
        full_size=replay_length,
    )

    sample_tree["next_observation"] = sample_tree["observation"].new_array(
        padding=0,
        inherit_full_size=True,
    )

    sample_tree["observation"][0] = obs_space.sample()
    metadata.example_obs_batch = sample_tree["observation"][0]

    device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device}, allow_val_change=True)
    device = torch.device(device)

    transformer_block_factory = instantiate(
        config.model.transformer_block,
        embedding_size=config.model.tokenizer.embedding_size,
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

    encoder_pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    tokenizer = instantiate(config.model.tokenizer, _convert_="partial")

    gpt_encoder = instantiate(
        config.model.gpt_encoder,
        transformer_encoder=transformer_encoder,
        pos_embedder=encoder_pos_embedder,
    )

    decoder_pos_embedder = instantiate(config.model.pos_embedder, _convert_="partial")
    masked_decoder = GPTDecoder(
        transformer_decoder=transformer_decoder,
        pos_embedder=decoder_pos_embedder,
    )

    mae_prediction_head = PredictionHead(
        dim=config.model.tokenizer.embedding_size,
        group_size=config.model.tokenizer.group_size,
        point_dim=config.model.prediction_head.point_dim,
    )

    finetune_encoder = FinetuneEncoder(
        transformer_encoder=transformer_encoder, pos_embedder=encoder_pos_embedder
    )
    aux_gpt = AuxGPT(
        gpt_encoder=gpt_encoder,
        gpt_decoder=masked_decoder,
        prediction_head=mae_prediction_head,
    )
    mlp_input_size = aux_gpt.dim

    pi_model = instantiate(
        config.model.pi_mlp_head,
        input_size=mlp_input_size,
        action_size=n_actions,
        action_space=action_space,
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
            "tokenizer": tokenizer,
            "encoder": finetune_encoder,
            "aux": aux_gpt,
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
    )

    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=replay_length,
        replay_batch_size=config.algo.batch_size,
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=batch_transform,
    )

    optimizer_conf = config.get("optimizer", {})
    optimizer_conf = OmegaConf.to_container(
        optimizer_conf, resolve=True, throw_on_missing=True
    )
    per_module_conf = optimizer_conf.pop("per_module", {})  # type: ignore

    pi_optimizer = torch.optim.Adam(
        [
            {
                "params": agent.model["pi"].parameters(),
                **per_module_conf.get("pi", {}),
            }
        ],
        **optimizer_conf,
    )
    encoder = agent.model["encoder"]

    # encoder_additional_params = encoder.get_additional_parameters()

    aux_optimizer = torch.optim.Adam(
        [
            {
                "params": agent.model["tokenizer"].parameters(),
                **per_module_conf.get("aux", {}),
            },
            {
                "params": agent.model["aux"].parameters(),
                **per_module_conf.get("aux", {}),
            },
        ],
        **optimizer_conf,
    )
    q_optimizer = torch.optim.Adam(
        [
            {
                "params": agent.model["tokenizer"].parameters(),
                **per_module_conf.get("tokenizer", {}),
            },
            {
                "params": agent.model["encoder"].parameters(),
                **per_module_conf.get("encoder", {}),
            },
            # {
            #     "params": encoder_additional_params,
            #     **per_module_conf.get("encoder", {}),
            # },
            # {
            #     "params": agent.model["aux"].parameters(),
            #     **per_module_conf.get("encoder", {}),
            # },
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
    )

    if gamma := config.get("lr_scheduler_gamma") is not None:
        pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=gamma)
        q_scheduler = torch.optim.lr_scheduler.ExponentialLR(q_optimizer, gamma=gamma)
        lr_schedulers = [pi_scheduler, q_scheduler]
    else:
        lr_schedulers = None

    algorithm = instantiate(
        config.algo,
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        pi_optimizer=pi_optimizer,
        q_optimizer=q_optimizer,
        aux_optimizer=aux_optimizer,
        lr_schedulers=lr_schedulers,
        _convert_="partial",
    )

    eval_cage_kwargs = dict(
        EnvClass=env_factory,
        env_kwargs={"add_obs_to_info_dict": True},
        TrajInfoClass=TrajInfoClass,
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
        sample_tree=eval_sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        output_dir=Path(config.video_path),
        video_length=config.env.max_episode_steps,
        use_wandb=True,
    )
    recording_schedule = RecordingSchedule(video_recorder, trigger="on_eval")

    eval_sampler = EvalSampler(
        max_traj_length=config.env.max_episode_steps,
        max_trajectories=config.eval.max_trajectories,
        envs=eval_cages,
        agent=agent,
        sample_tree=eval_sample_tree,
        step_transforms=[video_recorder],
    )

    # create runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=eval_sampler,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
        callbacks=[recording_schedule],
    )

    try:
        yield runner

    finally:
        eval_sampler.close()
        eval_sample_tree.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for eval_cage in eval_cages:
            eval_cage.close()
        for cage in cages:
            cage.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_aux_sac")
def main(config: DictConfig) -> None:
    if config.use_slurm:
        os.system("wandb enabled")
        tmp = Path(os.environ.get("TMPDIR"))  # type: ignore
        os.environ["WANDB_DIR"] = os.environ["TMPDIR"] + "/wandb"
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    run = wandb.init(
        project="pc_rl",
        tags=config.tags,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,
    )

    if config.use_slurm:
        video_path = (
            tmp
            / config.video_path
            / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
        )
    else:
        video_path = (
            Path(config.video_path)
            / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
        )
    config.update({"video_path": video_path})

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
        },  # type: ignore
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        model_save_path=Path("model.pt"),
        # verbosity=Verbosity.DEBUG,
    )

    with build(config) as runner:
        runner.run()

    logger.close()
    run.finish()  # type: ignore


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
