# pylint: disable=not-callable
# pylint: disable=no-member

import torch


class Agent(object):
    def __init__(self, env, planner, model, action_size, device="cpu"):
        self.env = env
        self.planner = planner
        self.model = model
        self.action_size = action_size
        self.device = device

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

    def run_episode(self, buffer=None, action_noise=None, render=False):
        with torch.no_grad():
            hidden, state = self.model.init_hidden_state(1)
            action = torch.zeros(1, self.action_size, device=self.device)

            obs = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                encoded_obs = self.model.encode_obs(obs)
                encoded_obs = encoded_obs.unsqueeze(0)
                action = action.unsqueeze(0)

                hidden, _, _, _, _, _, state = self.model.perform_rollout(
                    hidden, state, action, encoded_obs
                )

                hidden = hidden.squeeze(0)
                state = state.squeeze(0)
                action = self.planner(hidden, state)

                if action_noise is not None:
                    action = action + action_noise * torch.randn_like(action)

                next_obs, reward, done = self.env.step(action[0].cpu())
                total_reward += reward.item()

                if buffer is not None:
                    buffer.add(obs, action, reward, done)

                obs = next_obs

                if render:
                    self.env.render()
                if done:
                    break

            self.env.close()

        if buffer is None:
            return total_reward
        else:
            return total_reward, buffer

