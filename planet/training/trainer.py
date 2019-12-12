# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from planet import tools


class Trainer(object):
    def __init__(
        self,
        encoder,
        decoder,
        dynamics,
        reward_model,
        hidden_size,
        state_size,
        free_nats=None,
        device="cpu",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.dynamics = dynamics
        self.reward_model = reward_model

        self.hidden_size = hidden_size
        self.state_size = state_size
        self.device = device

        if free_nats is not None:
            self.free_nats = torch.full((1,), free_nats).to(self.device)
        else:
            self.free_nats = None

    def train_batch(self, buffer, batch_size, seq_len):
        obs, acts, rews, non_terms = buffer.sample(batch_size, seq_len)
        obs, acts, rews, non_terms = self._shift_sequences(obs, acts, rews, non_terms)

        hiddens, prior_mus, prior_stds, _, post_mus, post_stds, post_states = self._perform_rollout(
            obs, acts, non_terms
        )

        decoded_obs = self._bottle(self.decoder, (hiddens, post_states))
        decoded_reward = self._bottle(self.reward_model, (hiddens, post_states))

        posterior = Normal(post_mus, post_stds)
        prior = Normal(prior_mus, prior_stds)

        obs_loss = self._observation_loss(decoded_obs, obs)
        reward_loss = self._reward_loss(decoded_reward, rews)
        kl_loss = self._kl_loss(posterior, prior)

        return obs_loss, reward_loss, kl_loss

    def get_obs_rollout(self, buffer, seq_len):
        with torch.no_grad():
            obs, acts, rews, non_terms = buffer.sample(1, seq_len)
            obs, acts, rews, non_terms = self._shift_sequences(
                obs, acts, rews, non_terms
            )

            hiddens, _, _, _, _, _, post_states = self._perform_rollout(
                obs, acts, non_terms
            )

            decoded_obs = self._bottle(self.decoder, (hiddens, post_states))

        return decoded_obs, obs

    def get_params(self):
        return (
            list(self.decoder.parameters())
            + list(self.encoder.parameters())
            + list(self.reward_model.parameters())
            + list(self.dynamics.parameters())
        )

    def _perform_rollout(self, obs, actions, non_terminals=None):
        """ (seq_len, batch_size, *dims) """
        batch_size = obs.size(1)

        init_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        init_state = torch.zeros(batch_size, self.state_size).to(self.device)

        encoded_obs = self._bottle(self.encoder, (obs,))
        return self.dynamics(
            init_hidden,
            init_state,
            actions,
            obs=encoded_obs,
            non_terminals=non_terminals,
        )

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

    def _shift_sequences(self, obs, actions, rewards, non_terminals):
        obs = obs[1:]
        actions = actions[:-1]
        rewards = rewards[:-1]
        non_terminals = non_terminals[:-1]
        return obs, actions, rewards, non_terminals

    def _bottle(self, f, x_tuple):
        """ loops over the first dim of `x` (sequence) and applies f """
        x_sizes = tuple(map(lambda x: x.size(), x_tuple))
        y = f(
            *map(
                lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)
            )
        )
        y_size = y.size()
        return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])

