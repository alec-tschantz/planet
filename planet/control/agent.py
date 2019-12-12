# pylint: disable=not-callable
# pylint: disable=no-member

import torch


class Agent(object):
    def __init__(self, env, planner=None):
        self.env = env
        self.planner = planner

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_obs, reward, done = self.env.step(action)
                buffer.add(obs, action, reward, done)
                obs = next_obs
                if done:
                    break
        self.env.close()
        return buffer
