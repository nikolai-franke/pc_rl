import gymnasium as gym
import mani_skill2.envs

env = gym.make(
    "PickCube-v0",
    obs_mode="point_cloud",
    control_mode="pd_joint_delta_pos",
    render_mode="human",
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, reset_info = env.reset(seed=0)  # reset with a seed for randomness
terminated, truncated = False, False
while not terminated and not truncated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # a display is required to render
env.close()
