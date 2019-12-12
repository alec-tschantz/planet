import numpy as np

from planet import tools


def inspect_rollout(decoded_obs, obs, save_path=None):
    """ @TODO """
    seq_len = obs.size(0)

    """ suitable dims for plotting """
    seq_len, dim = _convert_seq_len(seq_len)

    decoded_obs = decoded_obs.squeeze(1)
    obs = obs.squeeze(1)

    imgs = []
    for t in range(seq_len):
        img = tools.obs_to_img(obs[t])
        imgs.append(img)

    for t in range(seq_len):
        img = tools.obs_to_img(decoded_obs[t])
        imgs.append(img)

    tools.plot_imgs(imgs, shape=(dim, dim), save_path=save_path)


def _convert_seq_len(seq_len):
    n = seq_len * 2
    dim = np.floor(np.sqrt(n))
    return int((dim ** 2) // 2), int(dim)
