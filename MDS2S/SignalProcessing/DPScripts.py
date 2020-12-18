import os
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import itertools
import tensorflow as tf

from config import Config as config

from .ymScripts.utils import visualize_signals

# from .EDA import autoEDA
# from .EMD import autoEMD
# from .VMD import autoVMD


ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/all'


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
    test_times = n // (config.NUM_CLASSES * config.NUM_SENSORS)
    maps = {}
    for file_name in file_names:
        if file_name[-5] == '0':
            if '0' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['0'] = groups
            else:
                groups = maps['0']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
                maps['0'] = groups
        elif file_name[-5] == '1':
            if '1' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['1'] = groups
            else:
                groups = maps['1']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
                maps['1'] = groups
        elif file_name[-5] == '2':
            if '2' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['2'] = groups
            else:
                groups = maps['2']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
                maps['2'] = groups
        elif file_name[-5] == '3':
            if '3' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['3'] = groups
            else:
                groups = maps['3']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
                maps['3'] = groups
        elif file_name[-5] == '4':
            if '4' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['4'] = groups
            else:
                groups = maps['4']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
                maps['4'] = groups
        elif file_name[-5] == '5':
            if '5' not in maps:
                groups = [['Nan'] * config.NUM_SENSORS for _ in range(test_times)]
                pos = int(file_name[-7]) - 1
                groups[0][pos] = file_name
                maps['5'] = groups
            else:
                groups = maps['5']
                pos = int(file_name[-7]) - 1
                for i in range(test_times):
                    if groups[i][pos] is 'Nan':
                        groups[i][pos] = file_name
                        break
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
    time_domain_res = None
    freq_domain_res = None
    for file_name in inputs:
        lamb_data = scio.loadmat(os.path.join(ROOT_DIR, file_name))['exportData'][0][0][0][0]
        if time_domain_res is None:
            res = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD]
            time_domain_res = res.reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1]) * config.AMPLIFIER

            freq_domain_res = tf.abs(tf.signal.stft(res, frame_length=config.WINDOW,
                                                    frame_step=config.STEP_SIZE))
            # freq_width_padding = (config.SIGNAL_FREQ - freq_domain_res.shape[0])
            # freq_height_padding = (config.SIGNAL_FREQ - freq_domain_res.shape[1])
            freq_width_padding = 0
            freq_height_padding = 0
            freq_domain_res = tf.pad(tensor=freq_domain_res,
                                     paddings=[(0, freq_width_padding), (0, freq_height_padding)],
                                     constant_values=0)[..., tf.newaxis]
        else:
            temp_res = lamb_data[: config.SIGNAL_FREQ * config.SIGNAL_PERIOD]
            time_domain_temp_res = temp_res.reshape([config.SIGNAL_FREQ, config.SIGNAL_PERIOD, 1]) * config.AMPLIFIER

            freq_domain_temp_res = tf.abs(tf.signal.stft(temp_res, frame_length=config.WINDOW,
                                                         frame_step=config.STEP_SIZE))
            # freq_width_padding = (config.SIGNAL_FREQ - freq_domain_temp_res.shape[0])
            # freq_height_padding = (config.SIGNAL_FREQ - freq_domain_temp_res.shape[1])
            freq_width_padding = 0
            freq_height_padding = 0
            freq_domain_temp_res = tf.pad(tensor=freq_domain_temp_res,
                                          paddings=[(0, freq_width_padding), (0, freq_height_padding)],
                                          constant_values=0)[..., tf.newaxis]

            time_domain_res = np.concatenate([time_domain_res, time_domain_temp_res], axis=2)
            freq_domain_res = np.concatenate([freq_domain_res, freq_domain_temp_res], axis=2)

    return time_domain_res, freq_domain_res, label


def dataGenerator():
    file_names = os.listdir(ROOT_DIR)
    data = getDataPairs(file_names)

    inputs, labels = data[0], data[1]
    n = len(inputs)
    for i in itertools.count(1):
        idx = i % n
        yield prepareTrainData(inputs[idx], labels[idx])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = tf.data.Dataset.from_generator(dataGenerator, (tf.float32, tf.float32, tf.int16),
                                             (tf.TensorShape([config.SIGNAL_PERIOD, config.SIGNAL_FREQ, 3]),
                                              tf.TensorShape([None, None, 3]),
                                              tf.TensorShape([None])))
    dataset = dataset.batch(2).shuffle(12)
    for time_data, freq_data, label in dataset.take(2):
        # plt.imshow(time_data[0, :, :, 0])
        print("{}--{}--{}".format(time_data.shape, freq_data.shape, label))
        visualize_signals(freq_data.numpy(), ylim=(0, 10), show_batch=2, )
    plt.show()

    # for _, freq_data, _ in dataset.take(1):
    #     for i in range(2):
    #         d1 = freq_data[i, :, :, 2]
    #         log_spec = np.log(d1.numpy().T)
    #         height = log_spec.shape[0]
    #         X = np.arange(193 * 65, step=height + 1)
    #         Y = range(height)
    #         plt.pcolormesh(X, Y, log_spec)
    # plt.show()



