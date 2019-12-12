# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F


class Planner(nn.Module):
    def __init__(
        self,
        dynamics,
        reward_model,
        action_size,
        plan_horizon,
        optim_iters,
        candidates,
        top_candidates,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.reward_model = reward_model
        self.action_size = action_size
        self.plan_horizon = plan_horizon
        self.optim_iters = optim_iters
        self.candidates = candidates
        self.top_candidates = top_candidates

    def forward(self, hidden, state):
        batch_size = hidden.size(0)
        hidden_size = hidden.size(1)
        state_size = state.size(1)
        hidden = (
            hidden.unsqueeze(dim=1)
            .expand(batch_size, self.candidates, hidden_size)
            .reshape(-1, hidden_size)
        )
        state = (
            state.unsqueeze(dim=1)
            .expand(batch_size, self.candidates, state_size)
            .reshape(-1, state_size)
        )
        action_mean = torch.zeros(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )
        action_std_dev = torch.ones(
            self.plan_horizon, batch_size, 1, self.action_size, device=hidden.device
        )

        for _ in range(self.optim_iters):
            actions = action_mean + action_std_dev * torch.randn(
                self.plan_horizon,
                batch_size,
                self.candidates,
                self.action_size,
                device=action_mean.device,
            )
            actions = actions.view(
                self.plan_horizon, batch_size * self.candidates, self.action_size
            )

            hiddens, _, _, states = self.dynamics(hidden, state, actions)
            returns = (
                self.reward_model(
                    hiddens.view(-1, hidden_size), states.view(-1, state_size)
                )
                .view(self.plan_horizon, -1)
                .sum(dim=0)
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
