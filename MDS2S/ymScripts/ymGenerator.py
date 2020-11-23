import tensorflow as tf
import numpy as np
import scipy.io as scio
import math
import os


ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/1120'


def prepareTrainData(inputs, labels, config):
    """
    Read data from files, then reshape data from different sensors into [config.signal_freq, config.signal_period, 1]
    and concatenate them together to get [config.signal_freq, config.signal_period, num_sensors]
    :param inputs: file names, shape: [num_sensors]
    :param labels: damage  types, shape: 1
    :return: [config.signal_freq, config.signal_period, num_sensors], 1
    """
    res = None
    for file_name in inputs[0]:
        lamb_data = scio.loadmat(os.path.join(ROOT_DIR, file_name))['exportData'][0][0][0][0]
        if res is None:
            res = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD].\
                reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1])
        else:
            temp = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD].\
                reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1])
            res = np.concatenate([res, temp], axis=2)
    return res, labels


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.BATCH_SIE = config.BATCH_SIZE
        self.SHUFFLE = config.SHUFFLE
        self.data_ids = range(len(dataset[0]))

    def __len__(self):
        return math.ceil(len(self.dataset[0]) / self.BATCH_SIE)

    def __getitem__(self, idx):
        file_names = self.dataset[0]
        labels = self.dataset[1]
        b = 0
        while b < self.BATCH_SIE:
            pass
