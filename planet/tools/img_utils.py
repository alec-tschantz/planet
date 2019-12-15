# pylint: disable=not-callable
# pylint: disable=no-member

import numpy as np
import cv2
import torch
from torchvision.utils import save_image


def img_to_obs(rgb_array, bits=5):
    img = cv2.resize(rgb_array, (64, 64), interpolation=cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1)
    obs = torch.tensor(img, dtype=torch.float32)
    obs = preprocess_obs(obs, bits)
    obs = obs.unsqueeze(0)
    return obs


def obs_to_img(obs, postprocess=True, bits=5):
    obs = obs.permute(1, 2, 0)
    obs = obs.cpu().detach().numpy()
    if postprocess:
        obs = postprocess_obs(obs, bits)
    return obs


def preprocess_obs(obs, bits):
    obs = obs.div_(2 ** (8 - bits)).floor_().div_(2 ** bits).sub_(0.5)
    obs = obs.add_(torch.rand_like(obs).div_(2 ** bits))
    return obs


def postprocess_obs(obs, bits):
    return np.clip(
        np.floor((obs + 0.5) * 2 ** bits) * 2 ** (8 - bits), 0, 2 ** 8 - 1
    ).astype(np.uint8)


def write_video(frames, save_path):
    """ convert to (h, w, c) """
    frames = np.stack(frames, axis=0).transpose(0, 2, 3, 1)
    frames = np.multiply(frames, 255).clip(0, 255).astype(np.uint8)
    frames = frames[:, :, :, ::-1]
    _, h, w, _ = frames.shape
    writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h), True
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_imgs(frames, save_path):
    save_image(torch.as_tensor(frames[-1]), save_path)
