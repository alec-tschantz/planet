# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import pathlib
import argparse
from os import makedirs

import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import roboschool

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from planet.envs import GymEnv, DebugEnv
from planet.training import Buffer, Trainer
from planet.control import Agent, Planner
from planet.models import RSSModel
from planet import tools


def main(args):
    makedirs(args.data_path, exist_ok=True)

    env = DebugEnv()
    action_size = env.action_size[0]
    buffer = Buffer(action_size, args.pixels, device=args.device)

    rssm = RSSModel(
        action_size,
        args.hidden_size,
        args.state_size,
        args.embedding_size,
        args.node_size,
        device=args.device,
    )

    trainer = Trainer(rssm, free_nats=args.free_nats, device=args.device)
    optim = torch.optim.Adam(rssm.parameters(), lr=args.lr, eps=args.epsilon)

    planner = Planner(
        rssm,
        action_size,
        args.plan_horizon,
        args.optim_iters,
        args.candidates,
        args.top_candidates,
    )

    agent = Agent(env, planner, rssm, action_size, device=args.device)
    buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes)

    for episode in range(args.n_episodes):
        for epoch in range(args.n_train_epochs):
            obs_loss, reward_loss, kl_loss = trainer.train_batch(
                buffer, args.batch_size, args.seq_len
            )

            optim.zero_grad()
            (obs_loss + reward_loss + kl_loss).backward()
            optim.step()

            loss = (obs_loss + reward_loss + kl_loss).item()

            if epoch % args.log_every == 0:
                print("Train Epoch {} | [{:3f}]".format(epoch, loss))

        _, buffer = agent.run_episode(buffer, action_noise=args.action_noise)
        reward, buffer = agent.run_episode(buffer)
        print("Episode {} | Reward [{:3f}]".format(episode, reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pixels", type=bool, default=True)
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--state_size", type=int, default=10)
    parser.add_argument("--embedding_size", type=int, default=200)
    parser.add_argument("--node_size", type=int, default=50)
    parser.add_argument("--free_nats", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--plan_horizon", type=int, default=2)
    parser.add_argument("--optim_iters", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--top_candidates", type=int, default=10)
    parser.add_argument("--n_seed_episodes", type=int, default=2)
    parser.add_argument("--n_train_epochs", type=int, default=5)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--action_noise", type=float, default=0.3)
    args = parser.parse_args()
    main(args)

