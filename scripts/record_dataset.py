from pathlib import Path
import os
import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pc_rl.builder  # for hydra's instantiate


def build(config: DictConfig):
    env = instantiate(config.env, _convert_="object")
    return env


@hydra.main(version_base=None, config_path="../conf", config_name="record_dataset")
def main(config: DictConfig):
    env = build(config)
    save_path = Path(__file__).parent.parent.resolve() / config.save_path
    os.makedirs(save_path, exist_ok=True)
    finished = False
    obs_counter = 0
    while not finished:
        done = False
        env.reset()
        while not done:
            obs, _, terminated, truncated, *_ = env.step(env.action_space.sample())
            np.save(save_path / str(obs_counter).zfill(6), obs)
            # np.savetxt(save_path / (str(obs_counter).zfill(6) + ".csv"), obs, delimiter=",")
            done = terminated or truncated

            obs_counter += 1
            if obs_counter >= config.num_observations:
                finished = True
                break

if __name__ == "__main__":
    main()
