# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from planet import tools


class Trainer(object):
    def __init__(self, model=None, free_nats=None, device="cpu"):
        self.model = model
        self.device = device
        
        self.free_nats = free_nats
        if self.free_nats is not None:
            self.free_nats = torch.full((1,), free_nats).to(self.device)

    def train_batch(self, buffer, batch_size, seq_len):
        obs, acts, rews, non_terms = buffer.sample_and_shift(batch_size, seq_len)

        rollout = self.model.perform_obs_rollout(obs, acts, non_terms)
        hidden, prior_mu, prior_std, _, post_mu, post_std, post_states = rollout

        decoded_obs = self.model.decode_obs_seq(hidden, post_states)
        decoded_reward = self.model.decode_reward_seq(hidden, post_states)

        posterior = Normal(post_mu, post_std)
        prior = Normal(prior_mu, prior_std)

        obs_loss = self._observation_loss(decoded_obs, obs)
        reward_loss = self._reward_loss(decoded_reward, rews)
        kl_loss = self._kl_loss(posterior, prior)

        return obs_loss, reward_loss, kl_loss

    def _observation_loss(self, decoded_obs, obs):
        return (
            F.mse_loss(decoded_obs, obs, reduction="none")
            .sum(dim=(2, 3, 4))
            .mean(dim=(0, 1))
        )

    def _reward_loss(self, decoded_reward, reward):
        return F.mse_loss(decoded_reward, reward, reduction="none").mean(dim=(0, 1))

    def _kl_loss(self, posterior, prior):
        if self.free_nats is not None:
            return torch.max(
                kl_divergence(posterior, prior).sum(dim=2), self.free_nats
            ).mean(dim=(0, 1))
        else:
            return kl_divergence(posterior, prior).sum(dim=2).mean(dim=(0, 1))

