# pylint: disable=not-callable
# pylint: disable=no-member

import torch
from torchvision.utils import make_grid

def test_rollout(env, model, planner):

    model.eval()

    with torch.no_grad():
        obs = env.reset()
        done = False
        total_reward = 0
        frames = []
        hidden, state, action = model.init_hidden_state_action(1)

        while not done:
            encoded_obs = model.encode_obs(obs)
            encoded_obs = encoded_obs.unsqueeze(0)
            action = action.unsqueeze(0)
            rollout = model.perform_rollout(
                action, hidden=hidden, state=state, obs=encoded_obs
            )
            hidden = rollout["hiddens"].squeeze(0)
            state = rollout["posterior_states"].squeeze(0)
            action = planner(hidden, state)
            next_obs, reward, done = env.step(action[0].cpu())
            total_reward += reward.item()
            
            """ store frames """
            decoded_obs = model.decode_obs(hidden, state)
            cat = torch.cat([obs, decoded_obs], dim=3)
            grid = make_grid(cat + 0.5, nrow=5).cpu().numpy()
            frames.append(grid)
      
            obs = next_obs
        env.close()
        return frames

