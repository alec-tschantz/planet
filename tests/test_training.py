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
from planet.training import Buffer, Trainer, inspect_rollout
from planet.control import Agent
from planet.models import ConvEncoder, ConvDecoder, RewardModel, RecurrentDynamics
from planet import tools


def main(args):
    makedirs(args.data_path, exist_ok=True)

    if args.debug:
        env = DebugEnv()
    else:
        env = GymEnv(args.env_name, args.pixels, args.max_episode_len, device=args.device)

    agent = Agent(env)

    buffer = Buffer(env.action_size[0], args.pixels, device=args.device)
    buffer = agent.get_seed_episodes(buffer, args.n_seed_episodes)
    print(
        "Collected {} episodes [{} frames]".format(
            buffer.total_episodes, buffer.total_steps
        )
    )

    encoder = ConvEncoder(args.embedding_size).to(args.device)
    decoder = ConvDecoder(args.hidden_size, args.state_size, args.embedding_size).to(args.device)
    reward_model = RewardModel(args.hidden_size, args.state_size, args.node_size).to(args.device)
    dynamics = RecurrentDynamics(
        args.hidden_size,
        args.state_size,
        env.action_size[0],
        args.node_size,
        args.embedding_size,
    ).to(args.device)

    trainer = Trainer(
        encoder,
        decoder,
        dynamics,
        reward_model,
        args.hidden_size,
        args.state_size,
        free_nats=args.free_nats,
        device=args.device
    )
    optim = torch.optim.Adam(trainer.get_params(), lr=args.lr, eps=args.epsilon)

    for epoch in range(args.n_train_epochs):

        obs_loss, reward_loss, kl_loss = trainer.train_batch(
            buffer, args.batch_size, args.seq_len
        )

        optim.zero_grad()
        (obs_loss + reward_loss + kl_loss).backward()
        optim.step()

        total_loss = (obs_loss + reward_loss + kl_loss).item()
        print("Epoch {}: [{:3f}]".format(epoch, total_loss))

        if epoch % args.log_every == 0:
            decoded_obs, obs = trainer.get_obs_rollout(buffer, args.validate_seq_len)
            save_path = "{}/rollout_{}.png".format(args.data_path, epoch)
            inspect_rollout(decoded_obs, obs, save_path=save_path)
            save_path = "{}/metrics.png".format(args.data_path)
            trainer.plot_metrics(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--n_seed_episodes", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=11)
    parser.add_argument("--validate_seq_len", type=int, default=11)
    parser.add_argument("--embedding_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--state_size", type=int, default=30)
    parser.add_argument("--node_size", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--env_name", type=str, default="RoboschoolInvertedPendulum-v1")
    parser.add_argument("--pixels", type=bool, default=True)
    parser.add_argument("--max_episode_len", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--n_train_epochs", type=int, default=200)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--free_nats", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    main(args)

