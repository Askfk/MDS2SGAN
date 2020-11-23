import os
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf

from ..config import Config as config

from .EDA import autoEDA
from .EMD import autoEMD
from .VMD import autoVMD


ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/1120'


def getData(file_path):
    """
    Read data from mat file
    :param file_path: file path
    :return: [
        s1_1 data,
        ...,
        sampling_rate,
        sampling_points
    ]
    """
    res = []

    data = scio.loadmat(file_path)['exportData']

    s1_1 = data[0][0][0][0]

    sampling_rate = int(data[0][0][-1][0][0][1][0][0])
    number_data_points = int(data[0][0][-1][0][0][2][0][0])

    res.append(s1_1)
    res.append(sampling_rate)
    res.append(number_data_points)
    return res


def getParams(file_name):
    """
    Get data params according to the file name
    :param file_name: file name
    :return: [
        sampling_rate,
        sampling_points,
        wave_freq,
        wave_amplitude,
        gain
    ]
    """
    params = file_name.split('-')

    sampling_rate = int(params[1][:2])
    sampling_points = int(params[2])

    wave_freq = int(params[3][:3])
    wave_amplitude = int(params[4][:4])
    gain = int(params[5][:2])

    return [sampling_rate, sampling_points, wave_freq, wave_amplitude, gain]


def getDataPairs(file_names, map_format=False, one_hot=True):
    """
    re-order dataset file names according to the file names. File name with same damage type will be set into one list.
    :param file_names: File names, shape: [n]
    :param map_format: True return map format data pairs otherwise return list format
    :return: if maps: {"damage type": [file name pairs]}
                else: [[file name pairs], [damage types]]
    """
    n = len(file_names)
    sensors = n // (6 * 2)
    maps = {}
    for file_name in file_names:
        if file_name[-5] == '0':
            if '0' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['0'] = groups
            else:
                groups = maps['0']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['0'] = groups
        elif file_name[-5] == '1':
            if '1' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['1'] = groups
            else:
                groups = maps['1']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['1'] = groups
        elif file_name[-5] == '2':
            if '2' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['2'] = groups
            else:
                groups = maps['2']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['2'] = groups
        elif file_name[-5] == '3':
            if '3' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['3'] = groups
            else:
                groups = maps['3']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['3'] = groups
        elif file_name[-5] == '4':
            if '4' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['4'] = groups
            else:
                groups = maps['4']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['4'] = groups
        elif file_name[-5] == '5':
            if '5' not in maps:
                groups = [['Nan'] * sensors for _ in range(2)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['5'] = groups
            else:
                groups = maps['5']
                pos = int(file_name[-7]) - 1
                if groups[0][pos] is not 'Nan':
                    groups[1][pos] = file_name
                else:
                    groups[0][pos] = file_name
                maps['5'] = groups
    if map_format:
        return maps
    train_data_file_name = []
    train_data_labels = []
    for status in maps:
        pairs = maps[status]
        for pair in pairs:
            train_data_file_name.append(pair)
            train_data_labels.append(int(status))
    if one_hot:
        train_data_labels = tf.one_hot(train_data_labels, depth=config.NUM_CLASSES)
    return [train_data_file_name, train_data_labels]


def prepareTrainData(inputs, label):
    """
    Read data from files, then reshape data from different sensors into [config.signal_freq, config.signal_period, 1]
    and concatenate them together to get [config.signal_freq, config.signal_period, num_sensors]
    :param inputs: file names, shape: [num_sensors]
    :param label: damage  types, shape: 1
    :return: [config.signal_freq, config.signal_period, num_sensors], 1
    """
    res = None
    for file_name in inputs:
        lamb_data = scio.loadmat(os.path.join(ROOT_DIR, file_name))['exportData'][0][0][0][0]
        if res is None:
            res = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD].\
                reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1]) * config.AMPLIFIER
        else:
            temp = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD].\
                reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1]) * config.AMPLIFIER
            res = np.concatenate([res, temp], axis=2)
    return res, label


def dataGenerator():
    file_names = os.listdir(ROOT_DIR)
    data = getDataPairs(file_names)

    inputs, labels = data[0], data[1]
    n = len(inputs)
    for i in itertools.count(1):
        idx = i % n
        yield prepareTrainData(inputs[idx], labels[idx])


if __name__ == '__main__':
    import tensorflow as tf
    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.float32, tf.int16),
                                             (tf.TensorShape([config.SIGNAL_PERIOD, config.SIGNAL_FREQ, 3]),
                                              tf.TensorShape([None])))
    dataset = dataset.batch(2).shuffle(12)
    for data, label in dataset:
        print("{}---{}".format(data.shape, label))


    # file_names = os.listdir(ROOT_DIR)
    # data = getDataPairs(file_names)
    # print(data)

