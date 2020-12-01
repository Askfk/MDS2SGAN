import math
import random
import shutil
from distutils.version import LooseVersion

import skimage.color
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
import warnings
import scipy
import urllib.request
import matplotlib.pyplot as plt
from ..config import Config

WEIGHTS_URL = ''
DATASET_URL = "https://github.com/Askfk/MDS2SGAN/releases/download/1/1120.zip"


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("Shape: {:20}   ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}   max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}   max: {:10}".format("", ""))

        text += "  {}".format(array.dtype)
    print(text)


def download_trained_weights(cocovg_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + cocovg_model_path + " ...")
    with urllib.request.urlopen(WEIGHTS_URL) as resp, open(cocovg_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def download_dataset(dataset_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading dataset to " + dataset_path + " ...")
    with urllib.request.urlopen(DATASET_URL) as resp, open(dataset_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading dataset!")


def visualize_signals(signals, ax=None, figsize=(16, 16)):
    """
    Visualize signals sets by subpot
    :param signals:
    :return:
    """

    n = signals.shape[0]
    plt.figure(figsize=figsize)
    if not ax:
        _, ax = plt.subplots(n, 1)

    for i in range(n):
        pass


def visualize_original_and_decomposed_modals(multi, single, show_batch=1, figsize=(25, 16), save_path=None,
                                             num_modals=Config.NUM_MODALS, denosing=None, ylim=(-0.8, 0.8)):
    """
    Visualize the original multi-modals signal and its corresponding single-modal signals

    :param ylim:
    :param denosing:
    :param num_modals:
    :param multi: multi-modal signal
    :param single: single-modal signals
    :param show_batch:
    :param ax:
    :param figsize:
    :param save_path: the path to save results
    """

    batch_size = min(show_batch, multi.shape[0])
    signal_nums = multi.shape[-1]
    figure, ax = plt.subplots(num_modals + 2, signal_nums, figsize=figsize)

    for i in range(batch_size):
        original_signal = multi[i]
        decomposed_signals = single[i]
        for j in range(signal_nums):
            sum_signal = tf.zeros_like(original_signal[:, :, j]).numpy().flatten()
            os = original_signal[:, :, j].numpy().flatten()
            if denosing:
                os = denosing(os).out
            ax[0, j].plot(os)
            ax[0, j].set_title("Original_{}".format(j + 1))
            ax[0, j].set_ylim(ylim)
            for n in range(num_modals):
                ds = decomposed_signals[:, :, n + j * num_modals].numpy().flatten()
                if denosing:
                    ds = denosing(ds).out
                sum_signal += ds
                ax[2 + n, j].plot(ds)
                ax[2 + n, j].set_title("Decomposed_{}_{}".format(j + 1, n + 1))
                ax[2 + n, j].set_ylim(ylim)
            error = os - sum_signal
            ax[1, j].plot(sum_signal, color='m')
            ax[1, j].plot(error, color='c')
            ax[1, j].set_title('sum_error_{}'.format(j+1))
            ax[1, j].set_ylim(ylim)
        if save_path:
            pass

    return figure, ax


if __name__ == '__main__':
    import tensorflow as tf

    original = tf.random.uniform([2, 96, 96, 3])
    decomposed = tf.random.uniform([2, 96, 96, 12])
    visualize_original_and_decomposed_modals(original, decomposed)