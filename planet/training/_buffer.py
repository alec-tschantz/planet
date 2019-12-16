# pylint: disable=not-callable
# pylint: disable=no-member

import os

import numpy as np
import torch

from planet import tools


class DataBuffer(object):
    def __init__(self, logdir, action_size, buffer_size=10 ** 6, bits=5, device="cpu"):

        self.action_size = action_size
        self.logdir = logdir
        self.buffer_size = buffer_size
        self.bits = bits
        self.device = device

        self.obs_path = os.path.join(logdir, "obs.dat")
        self.acts_path = os.path.join(logdir, "acts.dat")
        self.rews_path = os.path.join(logdir, "rews.dat")
        self.terms_path = os.path.join(logdir, "terms.dat")

        self.obs_shape = (buffer_size, 3, 64, 64)
        self.acts_shape = (buffer_size, action_size)
        self.rews_shape = (buffer_size,)
        self.terms_shape = (buffer_size, 1)

        self._create_buffer(self.obs_path, self.obs_shape, np.uint8)
        self._create_buffer(self.acts_path, self.acts_shape, np.float32)
        self._create_buffer(self.rews_path, self.rews_shape, np.float32)
        self._create_buffer(self.terms_path, self.terms_shape, np.float32)

        self.idx = 0
        self.full = False
        self.total_steps = 0
        self.total_episodes = 0

    def add(self, obs, action, reward, done):
        self._add_observation(obs)
        self._add_action(action)
        self._add_reward(reward)
        self._add_non_term(done)

        self.idx = (self.idx + 1) % self.buffer_size
        self.full = self.full or self.idx == 0
        self.total_steps = self.total_steps + 1
        if done:
            self.total_episodes = self.total_episodes + 1

    def sample(self, batch_size, seq_len):
        idxs = [self._get_sequence_idxs(seq_len) for _ in range(batch_size)]
        idxs = np.array(idxs)
        batch = self._get_batch(idxs, batch_size, seq_len)
        batch = [torch.as_tensor(item).to(device=self.device) for item in batch]
        return batch

    def sample_and_shift(self, batch_size, seq_len):
        obs, actions, rewards, non_terminals = self.sample(batch_size, seq_len)
        return self._shift_sequences(obs, actions, rewards, non_terminals)

    def _create_buffer(self, path, shape, dtype):
        f = np.memmap(path, dtype, mode="w+", shape=shape)
        del f

    def _get_buffer(self, path, shape, dtype):
        return np.memmap(path, dtype, mode="r+", shape=shape)

    def _read_buffer(self, path, shape, dtype):
        return np.memmap(path, dtype, mode="r", shape=shape)

    def _add_observation(self, obs):
        obs = obs.cpu().numpy()
        obs = tools.postprocess_obs(obs, self.bits)
        f = self._get_buffer(self.obs_path, self.obs_shape, np.uint8)
        f[self.idx] = obs
        del f

    def _add_action(self, action):
        f = self._get_buffer(self.acts_path, self.acts_shape, np.float32)
        f[self.idx] = action.cpu().numpy()
        del f

    def _add_reward(self, reward):
        f = self._get_buffer(self.rews_path, self.rews_shape, np.float32)
        f[self.idx] = reward
        del f

    def _add_non_term(self, done):
        f = self._get_buffer(self.terms_path, self.terms_shape, np.float32)
        f[self.idx] = not done
        del f

    def _get_batch(self, idxs, batch_size, seq_len):
        vec_idxs = idxs.transpose().reshape(-1)
        
        obs_f = self._read_buffer(self.obs_path, self.obs_shape, np.uint8)
        obs = torch.as_tensor(obs_f[vec_idxs].astype(np.float32))
        obs = tools.preprocess_obs(obs, self.bits)
        obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
        del obs_f

        acts_f = self._read_buffer(self.acts_path, self.acts_shape, np.float32)
        actions = acts_f[vec_idxs].reshape(seq_len, batch_size, -1)
        del acts_f

        rews_f = self._read_buffer(self.rews_path, self.rews_shape, np.float32)
        rewards = rews_f[vec_idxs].reshape(seq_len, batch_size)
        del rews_f

        terms_f = self._read_buffer(self.terms_path, self.terms_shape, np.float32)
        non_terminals = terms_f[vec_idxs].reshape(seq_len, batch_size, 1)
        del terms_f

        return obs, actions, rewards, non_terminals

    def _get_sequence_idxs(self, seq_len):
        valid = False
        while not valid:
            max_idx = self.buffer_size if self.full else self.idx - seq_len
            start_idx = np.random.randint(0, max_idx)
            idxs = np.arange(start_idx, start_idx + seq_len) % self.buffer_size
            valid = not self.idx in idxs[1:]
        return idxs

    def _shift_sequences(self, obs, actions, rewards, non_terminals):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        non_terminals = non_terminals[:-1]
        return obs, actions, rewards, non_terminals

    @property
    def current_size(self):
        return self.total_steps

    @property
    def current_episodes(self):
        return self.total_episodes

