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


def visualize_multi_and_single_modals(multi, single, ax=None, figsize=(16, 16), save_path=None):
    """
    Visualize the original multi-modals signal and its corresponding single-modal signals

    :param multi: multi-modal signal
    :param single: single-modal signals
    :param ax:
    :param figsize:
    :param save_path: the path to save results
    """

    batch_size = multi.shape[0]
    signal_nums = multi.shape[-1]

