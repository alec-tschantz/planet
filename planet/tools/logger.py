# pylint: disable=not-callable
# pylint: disable=no-member

import os
import json
import cv2
import pickle

import numpy as np
import torch


class Logger(object):
    def __init__(self, logdir, model, optim, buffer):
        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.buffer = buffer

        self.modeldir = os.path.join(self.logdir, "models")
        self.videodir = os.path.join(self.logdir, "videos")
        self.datadir = os.path.join(self.logdir, "data")

        os.makedirs(self.modeldir, exist_ok=True)
        os.makedirs(self.videodir, exist_ok=True)
        os.makedirs(self.datadir, exist_ok=True)

        print("Creating a new _logdir_ at {}".format(self.logdir))

    def create_log(self):
        print("Creating a new _logdir_ at {}".format(self.logdir))

    def load_log(self):
        print("Loading _dir_ from {}".format(self.logdir))
        with open(self.args_path) as json_file:
            self.args = json.load(json_file)
        self.load_models()

    """
    def load_models(self):
        model_dicts = torch.load($PATH)
        self.model.load_state_dict(model_dicts)
        self.optim.load_state_dict(model_dicts["optim"])
    """

    def checkpoint(self, episode):
        save_dict = self.model.get_save_dict()
        save_dict["optim"] = self.optim.state_dict()
        path = os.path.join(self.modeldir, "model_{}.pth".format(episode))
        torch.save(save_dict, path)

        path = os.path.join(self.datadir, "buffer_{}.pth".format(episode))
        with open(path, "wb") as pickle_file:
            pickle.dump(self.buffer, pickle_file)

    def save_video(self, frames, episode):
        path = os.path.join(self.videodir, "vid_{}.mp4".format(episode))
        frames = np.stack(frames, axis=0).transpose(0, 2, 3, 1)
        frames = np.multiply(frames, 255).clip(0, 255).astype(np.uint8)
        frames = frames[:, :, :, ::-1]
        _, h, w, _ = frames.shape
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h), True
        )
        for frame in frames:
            writer.write(frame)
        writer.release()

