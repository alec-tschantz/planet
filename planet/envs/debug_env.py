# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import torch
import gym

from planet import tools


class DebugEnv(object):
    def __init__(self, max_episode_len=200, seed=None, bits=5, device="cpu"):
        self.max_episode_len = max_episode_len
        self.seed = seed
        self.device = device
        self.bits = bits

        self.done = False
        self.t = None

    def reset(self):
        self.t = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        reward = np.random.uniform(0, 1)
        self.t += 1
        done = self.t >= self.max_episode_len
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        return self._get_observation(), reward, done

    def sample_action(self):
        return torch.from_numpy(self.action_space.sample()).to(self.device)

    def render(self):
        pass

    def close(self):
        pass

    @property
    def state_size(self):
        return self.observation_space.shape

    @property
    def obs_size(self):
        return (3, tools.IMG_SIZE, tools.IMG_SIZE)

    @property
    def action_size(self):
        return self.action_space.shape

    @property
    def observation_space(self):
        low = np.zeros([tools.IMG_SIZE, tools.IMG_SIZE, 3], dtype=np.float32)
        high = np.ones([tools.IMG_SIZE, tools.IMG_SIZE, 3], dtype=np.float32)
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        low = -np.ones([5], dtype=np.float32)
        high = np.ones([5], dtype=np.float32)
        return gym.spaces.Box(low, high)

    def _get_observation(self):
        rgb_array = self.observation_space.sample()
        obs = tools.img_to_obs(rgb_array, self.bits)
        return obs.to(self.device)
