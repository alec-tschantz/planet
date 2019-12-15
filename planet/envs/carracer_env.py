# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import gym
import torch

from planet import tools

class CarRacingEnv(object):
    def __init__(
        self,
        env_name="CarRacing-v0",
        pixels=False,
        max_episode_len=np.inf,
        action_repeat=1,
        bits=5,
        seed=None,
        device="cpu",
    ):

        self._env = gym.make("CarRacing-v0")
        self.pixels = False
        self.max_episode_len = max_episode_len
        self.action_repeat = action_repeat
        self.bits = bits
        self.seed = seed
        self.device = device

        self.done = False
        self.t = None
        self._state = None
        if seed is not None:
            self._env.seed(seed)

    def reset(self):
        self.t = 0
        self.done = False
        self._state = self._env.reset()
        self._env.env.viewer.window.dispatch_events()
        return self._get_observation()

    def step(self, action):
        action = action.cpu().detach().numpy()
        reward = 0

        for _ in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            self._env.env.viewer.window.dispatch_events()
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_len
            if done:
                self.done = True
                break

        self._state = state
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        return self._get_observation(), reward, done

    def sample_action(self):
        return torch.from_numpy(self._env.action_space.sample()).to(self.device)

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def _get_observation(self):
        obs = tools.img_to_obs(self._state, self.bits)
        return obs.to(self.device)

    @property
    def state_size(self):
        return (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape