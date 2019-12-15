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

        self.model_path = os.path.join(self.logdir, "models")
        self.video_path = os.path.join(self.logdir, "videos")
        self.args_path = os.path.join(self.logdir, "args.json")
        self.info_path = os.path.join(self.logdir, "exp_info.json")

        if os.path.exists(self.logdir):
            self.load_log()
        else:
            self.create_log()

    def create_log(self):
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)

        self.args = vars(self.args)
        self.info = {"episode": 0}

        with open(self.args_path, "w") as json_file:
            json.dump(self.args, json_file)

        with open(self.info_path, "w") as json_file:
            json.dump(self.info, json_file)

    def load_log(self):
        with open(self.args_path) as json_file:
            self.args = json.load(json_file)

        with open(self.info_path) as json_file:
            self.info = json.load(json_file)

        self.load_models()

    def checkpoint(self, episode):
        path = os.path.join(self.model_path, "model.pth")
        save_dict = self.model.get_save_dict()
        save_dict["optim"] = self.optim.state_dict()
        torch.save(save_dict, path)
        
        with open(self.args_path, "w") as json_file:
            json.dump(self.args, json_file)

        with open(self.info_path, "w") as json_file:
            json.dump(self.info, json_file)

    def set_data(self, episode):
        self.info["episode"] = episode

    def save_info(self, episode):
        json_f = json.dumps(self.exp_info)
        f = open(self.info_path, "w")
        f.write(json_f)
        f.close()

    def save_video(self, frames, episode):
        path = os.path.join(self.video_path, "vid_{}.mp4".format(episode))
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

    def load_models(self):
        if os.path.exists(self.model_path):
            model_path = os.path.join(self.model_path, "model.pth")
            model_dicts = torch.load(model_path)
            self.model.load_state_dict(model_dicts)
            self.optim.load_state_dict(model_dicts["optim"])

    @property
    def episode(self):
        return self.info["episode"]

