from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env

import wandb


def main():
    run = wandb.init(
        project="pc_rl",
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,
    )
    env_id = "LiftCube-v0"
    obs_mode = "rgbd"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "normalized_dense"
    num_envs = 20  # recommended value for Google Colab. If you have more cores and a more powerful GPU you can increase this
    env: VecEnv = make_vec_env(
        env_id,
        num_envs,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
    )
    obs, _ = env.reset(seed=0)
    print("Base Camera RGB:", obs["image"]["base_camera"]["rgb"].shape)
    print("Base Camera RGB device:", obs["image"]["base_camera"]["rgb"].device)
    for _ in range(10_000):
        env.step(env.action_space.sample())
    env.close()
    run.finish()


if __name__ == "__main__":
    main()
