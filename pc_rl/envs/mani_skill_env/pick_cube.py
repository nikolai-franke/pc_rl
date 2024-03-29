import numpy as np
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.registration import register_env


@register_env("PickCube-v1", max_episode_steps=200)
class PickCube(PickCubeEnv):
    goal_thresh = 0.05
    min_goal_dist = 0.1

    def __init__(
        self,
        *args,
        obj_init_rot_z=True,
        is_grasped_reward=1.0,
        always_target_dist_reward=False,
        **kwargs
    ):
        super().__init__(*args, obj_init_rot_z=obj_init_rot_z, **kwargs)
        self.was_grasped = False
        self.old_dist_from_cube = -1.0
        self.old_dist_from_target = -1.0
        self.is_grasped_reward = is_grasped_reward
        self.always_target_dist_reward = always_target_dist_reward

    # def _load_actors(self):
    #     self.cube_half_size = np.array([0.015] * 3, np.float32)
    #     return super()._load_actors()

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 100.0

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)

        if not is_grasped:
            tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
            tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
            reaching_reward = -self.is_grasped_reward - np.tanh(5 * tcp_to_obj_dist)
            reward += reaching_reward
            # if self.old_dist_from_cube != -1.0:
            #     reward += self.old_dist_from_cube - tcp_to_obj_dist
            # self.old_dist_from_cube = tcp_to_obj_dist

        if is_grasped or self.always_target_dist_reward:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = -np.tanh(5 * obj_to_goal_dist)
            reward += place_reward
            # if self.old_dist_from_target != -1.0:
            #     reward += self.old_dist_from_target - obj_to_goal_dist
            # self.old_dist_from_target = obj_to_goal_dist

        return reward
