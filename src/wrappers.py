import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


class AngleBalanceWrapper(Wrapper):
    def __init__(self, env: gym.Env, target_angle_deg: float) -> None:
        super().__init__(env)
        self.target_angle_rad = np.deg2rad(target_angle_deg)

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)

        theta = np.arctan2(obs[1], obs[0])
        theta_dot = obs[2]

        new_reward = -(
            (theta - self.target_angle_rad) ** 2
            + 0.1 * theta_dot**2
            + 0.001 * action[0] ** 2
        )

        return obs, new_reward, term, trunc, info
