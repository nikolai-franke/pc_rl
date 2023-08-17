import multiprocessing as mp
import os
import sys
import traceback
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
import pc_rl.models.sac.q_and_pi_heads
import wandb
from pc_rl.agents.sac import PcSacAgent
from pc_rl.models.finetune_encoder import FinetuneEncoder


@contextmanager
def build(config: DictConfig):
    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    TrajInfo.set_discount(discount)
    CageCls = ProcessCage if parallel else SerialCage
    storage = "shared" if parallel else "local"

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
        storage=storage,
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
            "embedder": embedder,
            "encoder": finetune_encoder,
        }
    )

    # instantiate agent
    agent = PcSacAgent(
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
                    "params": agent.model["encoder"].parameters(),
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
        output_dir=Path(config.video_path),
        video_length=config.eval.max_traj_length,
        use_wandb=True,
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
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
    )

    try:
        yield runner

    finally:
        sampler.close()
        agent.close()
        eval_sampler.close()
        for cage in cages:
            cage.close()
        for eval_cage in eval_cages:
            eval_cage.close()
        sample_tree.close()
        eval_sample_tree.close()


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    mp.set_start_method("forkserver")
    # try...except block so we get error messages when using submitit
    try:
        run = wandb.init(
            project="pc_rl",
            tags=["continuous", "sac"],
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
            sync_tensorboard=True,  # auto-upload any values logged to tensorboard
            save_code=True,  # save script used to start training, git commit, and patch
            reinit=True,
        )

        if config.use_slurm:  # TODO: check if launcher starts with submitit
            os.system("wandb enabled")
            tmp = Path(os.environ.get("TMP"))  # type: ignore
            video_path = (
                tmp
                / config.video_path
                / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
            )
            num_gpus = HydraConfig.get().launcher.gpus_per_node
            gpu_id = HydraConfig.get().job.num % num_gpus
            config.update({"device": f"cuda:{gpu_id}"})
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
            model_save_path="model.pt",
            # verbosity=Verbosity.DEBUG,
        )

        with build(config) as runner:
            runner.run()

        run.finish()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
