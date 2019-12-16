# pylint: disable=not-callable
# pylint: disable=no-member

import os
import json
import cv2

import numpy as np
import torch


class Logger(object):
    def __init__(self, logdir, model, optim, args):

        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.args = args

        self.modeldir = os.path.join(self.logdir, "models")
        self.videodir = os.path.join(self.logdir, "videos")
        self.args_path = os.path.join(self.logdir, "args.json")
        self.info_path = os.path.join(self.logdir, "info.json")
        self.model_path = os.path.join(self.modeldir, "models.pth")

        if os.path.exists(self.logdir):
            self.load_log()
        else:
            self.create_log()

    def create_log(self):
        print("Creating a new _logdir_ at {}".format(self.logdir))

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        os.makedirs(self.videodir, exist_ok=True)

        self.args = vars(self.args)
        self.info = {"episode": 0}

        with open(self.args_path, "w") as json_file:
            json.dump(self.args, json_file)
        with open(self.info_path, "w") as json_file:
            json.dump(self.info, json_file)

    def load_log(self):
        print("Loading _dir_ from {}".format(self.logdir))
        with open(self.args_path) as json_file:
            self.args = json.load(json_file)
        with open(self.info_path) as json_file:
            self.info = json.load(json_file)
        self.load_models()

    def load_models(self):
        if os.path.exists(self.modeldir):
            print("Loading _models_ from {}".format(self.model_path))
            model_dicts = torch.load(self.model_path)
            self.model.load_state_dict(model_dicts)
            self.optim.load_state_dict(model_dicts["optim"])

    def checkpoint(self):
        save_dict = self.model.get_save_dict()
        save_dict["optim"] = self.optim.state_dict()
        torch.save(save_dict, self.model_path)

        with open(self.info_path, "w") as json_file:
            json.dump(self.info, json_file)

    def log_data(self, episode):
        self.info["episode"] = episode

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

    @property
    def episode(self):
        return self.info["episode"]

