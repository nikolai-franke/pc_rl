import numpy as np
from mani_skill2.envs.ms1.push_chair import PushChairEnv
from mani_skill2.utils.registration import register_env
from scipy.spatial import distance as sdist


@register_env("PushChair-v2", max_episode_steps=200)
class PushChair(PushChairEnv):
    goal_thresh = 0.05
    min_goal_dist = 0.1

    def compute_dense_reward(self, action: np.ndarray, info: dict, **kwargs):
        reward = 0

        # Compute distance between end-effectors and chair surface
        ee_coords = np.array(self.agent.get_ee_coords())  # [M, 3]
        chair_pcd = self._get_chair_pcd()  # [N, 3]

        # EE approach chair
        dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [M, N]
        dist_ees_to_chair = dist_ees_to_chair.min(1)  # [M]
        dist_ee_to_chair = dist_ees_to_chair.mean()
        log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
        reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

        # Keep chair standing
        chair_tilt = info["chair_tilt"]
        reward += -chair_tilt * 0.2

        # Penalize action
        # Assume action is relative and normalized.
        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        # Chair velocity
        # Legacy version uses full velocity instead of xy-plane velocity
        chair_vel = self.root_link.velocity[:2]
        chair_vel_norm = np.linalg.norm(chair_vel)
        disp_chair_to_target = self.root_link.get_pose().p[:2] - self.target_xy
        cos_chair_vel_to_target = sdist.cosine(disp_chair_to_target, chair_vel)
        chair_ang_vel_norm = info["chair_ang_vel_norm"]

        # Stage reward
        # NOTE(jigu): stage reward can also be used to debug which stage it is
        stage_reward = -10
        # -18 can guarantee the reward is negative
        dist_chair_to_target = info["dist_chair_to_target"]

        if chair_tilt < 0.2 * np.pi:
            # Chair is standing
            if dist_ee_to_chair < 0.1:
                # EE is close to chair
                stage_reward += 2
                if dist_chair_to_target <= 0.15:
                    # Chair is close to target
                    stage_reward += 2
                    # Try to keep chair static
                    reward += np.exp(-chair_vel_norm * 10) * 2
                    # Legacy: Note that the static condition here is different from success metric
                    if chair_vel_norm <= 0.1 and chair_ang_vel_norm <= 0.2:
                        stage_reward += 2
                else:
                    # Try to increase velocity along direction to the target
                    # Compute directional velocity
                    x = (1 - cos_chair_vel_to_target) * chair_vel_norm
                    reward += max(-1, 1 - np.exp(x)) * 2 - dist_chair_to_target * 2

        reward = reward + stage_reward

        # Update info
        info.update(
            dist_ee_to_chair=dist_ee_to_chair,
            action_norm=action_norm,
            chair_vel_norm=chair_vel_norm,
            cos_chair_vel_to_target=cos_chair_vel_to_target,
            stage_reward=stage_reward,
        )
        return reward