# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from torchvision.utils import make_grid


class Agent(object):
    def __init__(self, env, planner, model, device="cpu"):
        self.env = env
        self.model = model
        self.planner = planner
        self.device = device

    def test_rollout(self, buffer=None, action_noise=None, frames=False, render=False):
        self.model.eval()

        with torch.no_grad():
            obs = self.env.reset()
            done = False
            total_reward = 0
            if frames is not None:
                frames = []
            hidden, state, action = self.model.init_hidden_state_action(1)

            while not done:
                encoded_obs = self.model.encode_obs(obs)
                encoded_obs = encoded_obs.unsqueeze(0)
                action = action.unsqueeze(0)
                rollout = self.model.perform_rollout(
                    action, hidden=hidden, state=state, obs=encoded_obs
                )
                hidden = rollout["hiddens"].squeeze(0)
                state = rollout["posterior_states"].squeeze(0)

                action = self.planner(hidden, state)
                action = self._add_action_noise(action, action_noise)

                next_obs, reward, done = self.env.step(action[0].cpu())

                total_reward += reward.item()

                if frames is not None:
                    decoded_obs = self.model.decode_obs(hidden, state)
                    cat = torch.cat([obs, decoded_obs], dim=3)
                    grid = make_grid(cat + 0.5, nrow=5).cpu().numpy()
                    frames.append(grid)

                """ update buffer """
                if buffer is not None:
                    buffer.add(obs, action, reward, done)
                obs = next_obs

                if render:
                    self.render()

                if done:
                    break

            self.env.close()

            if buffer is None and frames is None:
                return total_reward
            elif buffer is None:
                return total_reward, frames
            elif frames is None:
                return total_reward, buffer
            else:
                return total_reward, buffer, frames

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

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action
