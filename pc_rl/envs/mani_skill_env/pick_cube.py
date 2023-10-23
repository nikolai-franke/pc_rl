import numpy as np
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.registration import register_env


@register_env("PickCube-v1", max_episode_steps=200)
class PickCube(PickCubeEnv):
    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        super().__init__(*args, obj_init_rot_z=obj_init_rot_z, **kwargs)
        self.was_grasped = False
        self.old_dist_from_cube = -1.0
        self.old_dist_from_target = -1.0

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 100.0

        tcp_to_obj_pos = self.obj.pose.p - self.tcp.pose.p
        tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_pos)
        reaching_reward = -np.tanh(5 * tcp_to_obj_dist)
        reward += reaching_reward
        if self.old_dist_from_cube != -1.0:
            reward += self.old_dist_from_cube - tcp_to_obj_dist
            self.old_dist_from_target = tcp_to_obj_dist

        is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        if is_grasped and not self.was_grasped:
            reward += 5.0
        if not is_grasped and self.was_grasped:
            reward -= 5.0
        self.was_grasped = is_grasped

        if is_grasped:
            obj_to_goal_dist = np.linalg.norm(self.goal_pos - self.obj.pose.p)
            place_reward = -np.tanh(5 * obj_to_goal_dist)
            reward += place_reward
            if self.old_dist_from_target != -1.0:
                reward += self.old_dist_from_target - obj_to_goal_dist
                self.old_dist_from_target = obj_to_goal_dist

        return reward
