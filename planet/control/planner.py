# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F


class Planner(nn.Module):
    def __init__(
        self, model, action_size, plan_horizon, optim_iters, candidates, top_candidates
    ):
        super().__init__()
        self.model = model
        self.action_size = action_size
        self.plan_horizon = plan_horizon
        self.optim_iters = optim_iters
        self.candidates = candidates
        self.top_candidates = top_candidates

    def forward(self, hidden, state):
        """ (batch_size, dim) """
        batch_size = hidden.size(0)
        hidden_size = hidden.size(1)
        state_size = state.size(1)

        """ (batch_size, candidates, hidden_size) """
        hidden = hidden.unsqueeze(dim=1)
        hidden = hidden.expand(batch_size, self.candidates, hidden_size)
        """ (batch_size * candidates, hidden_size) """
        hidden = hidden.reshape(-1, hidden_size)

        """ (batch_size, candidates, state_size) """
        state = state.unsqueeze(dim=1)
        state = state.expand(batch_size, self.candidates, state_size)
        """ (batch_size * candidates, state_size) """
        state = state.reshape(-1, state_size)

        """ (plan_horizon, batch_size, 1, action_size) """
        action_mean = torch.zeros(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )
        action_std_dev = torch.ones(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )

        """ optimise """
        for _ in range(self.optim_iters):

            """ get candidates """
            epsilon = torch.randn(
                self.plan_horizon,
                batch_size,
                self.candidates,
                self.action_size,
                device=action_mean.device,
            )
            actions = action_mean + action_std_dev * epsilon

            """ (plan_horizon, batch_size * candidates, action_size) """
            actions = actions.view(
                self.plan_horizon, batch_size * self.candidates, self.action_size
            )

            """ (plan_horizon, batch_size * candidates, dim) """
            _hiddens, _, _, _states = self.model.perform_rollout(hidden, state, actions)

            """ (plan_horizon * batch_size * candidates, dim) """
            _hiddens = _hiddens.view(-1, hidden_size)
            _states = _states.view(-1, state_size)

            returns = self.model.predict_reward(_hiddens, _states)
            """ (plan_horizon, batch_size * candidates * 1) """
            returns = returns.view(self.plan_horizon, -1)
            """ sum over time """
            returns = returns.sum(dim=0)

            """ fit new Gaussian """
            returns = returns.reshape(batch_size, self.candidates)
            _, topk = returns.topk(
                self.top_candidates, dim=1, largest=True, sorted=False
            )

            _, topk = returns.reshape(batch_size, self.candidates).topk(
                self.top_candidates, dim=1, largest=True, sorted=False
            )
            topk += self.candidates * torch.arange(
                0, batch_size, dtype=torch.int64, device=topk.device
            ).unsqueeze(dim=1)

            best_actions = actions[:, topk.view(-1)].reshape(
                self.plan_horizon, batch_size, self.top_candidates, self.action_size
            )
            action_mean, action_std_dev = (
                best_actions.mean(dim=2, keepdim=True),
                best_actions.std(dim=2, unbiased=False, keepdim=True),
            )

        return action_mean[0].squeeze(dim=1)
