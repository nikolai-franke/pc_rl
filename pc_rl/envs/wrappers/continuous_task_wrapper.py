import gymnasium as gym


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, False, truncated, info
