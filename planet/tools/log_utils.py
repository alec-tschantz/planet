# pylint: disable=not-callable
# pylint: disable=no-member

import os
import json
import cv2
import pickle

import numpy as np
import torch

MODEL_FILE = "model.pth"
BUFFER_FILE = "buffer.pth"
METRICS_FILE = "metrics.json"


def logdir_exists(logdir):
    return os.path.exists(logdir)


def init_dirs(logdir):
    videosdir = os.path.join(logdir, "videos")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(videosdir, exist_ok=True)


def load_buffer(logdir):
    buffer_path = os.path.join(logdir, BUFFER_FILE)
    return _load_pickle(buffer_path)


def load_model_dict(logdir):
    model_path = os.path.join(logdir, MODEL_FILE)
    return torch.load(model_path)


def load_metrics(logdir):
    path = os.path.join(logdir, METRICS_FILE)
    return _load_json(path)


def save_model(logdir, model, optim):
    path = os.path.join(logdir, MODEL_FILE)
    save_dict = model.get_save_dict()
    save_dict["optim"] = optim.state_dict()
    torch.save(save_dict, path)
    print("Saved _models_ at `{}`".format(path))


def save_metrics(logdir, metrics):
    path = os.path.join(logdir, METRICS_FILE)
    _save_json(path, metrics)
    print("Saved _metrics_ at path `{}`".format(path))


def save_buffer(logdir, buffer):
    path = os.path.join(logdir, BUFFER_FILE)
    _save_pickle(path, buffer)
    print("Saved _buffer_ at path `{}`".format(path))


def save_video(logdir, episode, frames):
    videodir = os.path.join(logdir, "videos")
    path = os.path.join(videodir, "vid_{}.mp4".format(episode))
    frames = np.stack(frames, axis=0).transpose(0, 2, 3, 1)
    frames = np.multiply(frames, 255).clip(0, 255).astype(np.uint8)
    frames = frames[:, :, :, ::-1]
    _, h, w, _ = frames.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def build_metrics():
    return {
        "episode": 0,
        "train_rewards": [],
        "test_rewards": [],
        "obs_loss": [],
        "reward_loss": [],
        "kl_loss": [],
    }


def _load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def _save_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file)


def _load_pickle(path):
    with open(path, "rb") as pickle_file:
        data = pickle.load(pickle_file)
    return data


def _save_pickle(path, obj):
    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file)
