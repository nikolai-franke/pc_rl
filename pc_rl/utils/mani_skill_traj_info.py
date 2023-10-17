from __future__ import annotations

from dataclasses import dataclass

from parllel.cages.traj_info import (ActionType, DoneType, EnvInfoType,
                                     ObsType, RewardType, TrajInfo)


@dataclass
class SofaTrajInfo(TrajInfo):
    Success: bool = False
    SuccessLength: int = 0

    def step(
        self,
        observation: ObsType,
        action: ActionType,
        reward: RewardType,
        terminated: DoneType,
        truncated: DoneType,
        env_info: EnvInfoType,
    ) -> None:
        super().step(observation, action, reward, terminated, truncated, env_info)
        self.Success = env_info["success"]
        if not self.Success:
            self.SuccessLength += 1
