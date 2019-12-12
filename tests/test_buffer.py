# pylint: disable=not-callable
# pylint: disable=no-member

import sys
import pathlib
import argparse
import unittest

import torch

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from planet.envs import DebugEnv
from planet.training import Buffer
from planet.control import Agent
from planet import tools


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.env = DebugEnv()
        self.agent = Agent(self.env)
        self.buffer = Buffer(self.env.action_size[0], True)

    def test_batch_dimensions(self):
        batch_size = 30
        seq_len = 15
        buffer = self.agent.get_seed_episodes(self.buffer, 10)
        obs, _, _, _ = buffer.sample(batch_size, seq_len)

        seq_dim = obs.size(0)
        batch_dim = obs.size(1)
        img_dim = obs.size(3)
        n_dims = len(list(obs.shape))
        self.assertEqual(seq_dim, seq_len)
        self.assertEqual(batch_dim, batch_size)
        self.assertEqual(img_dim, tools.IMG_SIZE)
        self.assertEqual(n_dims, 5)


if __name__ == "__main__":
    unittest.main()
