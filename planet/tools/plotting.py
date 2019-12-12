import numpy as np
import matplotlib.pyplot as plt


def plot_lines(lines, save_path=None):
    n_lines = len(lines) if isinstance(lines, list) else 1
    _, axes = plt.subplots(1, 1)

    if n_lines == 1:
        axes.plot(lines)
    else:
        for i in range(n_lines):
            axes.plot(lines[i])

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close("all")


def plot_imgs(imgs, shape=None, save_path=None):
    """ @TODO """
    n_img = len(imgs) if isinstance(imgs, list) else 1

    if shape is None:
        _, axes = plt.subplots(1, n_img)
    else:
        _, axes = plt.subplots(*shape)

    if n_img == 1:
        axes.imshow(imgs)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    else:
        if shape is not None:
            axes = axes.flatten()
        for i in range(len(axes)):
            if i < n_img:
                axes[i].imshow(imgs[i])
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close("all")

