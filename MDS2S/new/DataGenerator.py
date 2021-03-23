import tensorflow as tf
import os
import numpy as np
import scipy.io as scio
import itertools
import pywt

from EDA import WaveletDenoising


class DataGenerator(object):
    """
    Data generator class.
    Used with tf.dataset class to initialize dataset
    """

    def __init__(self, config):
        self.config = config

    def get_data_paris(self):
        data_pairs = []
        lamb_file_names = os.listdir(self.config.DATA_ROOT_DIR)
        for lfn in lamb_file_names:
            nth = lfn.split('-')[0]
            file_address = os.path.join(self.config.DATA_ROOT_DIR, lfn)
            tf_feats = []
            lambs = scio.loadmat(file_address)
            IMFs = []
            IMFs_all = np.load(os.path.join(self.config.IMFs_ROOT_DIR, nth + '.npy'), allow_pickle=True)
            final = {}
            # Generate diced wavelet time-frequency feats and IMFs
            for channel in self.config.CHANNELS:
                signal = lambs['s{}'.format(channel)][:, 0]
                signal = WaveletDenoising(signal).out[self.config.TIME_RANGE[0]: self.config.TIME_RANGE[1]]
                fc = pywt.central_frequency(self.config.W_TF)
                cparam = 2 * fc * self.config.TOTALSCALE
                scales = cparam / np.arange(self.config.TOTALSCALE, 1, -1)
                [cwtmatr, _] = pywt.cwt(signal, scales, self.config.W_TF,
                                        1.0 / self.config.SAMPLING_RATE)  # 连续小波变换
                cwtmatr = abs(cwtmatr)[self.config.FREQ_RANGE[0]: self.config.FREQ_RANGE[1], :]  # shape: [144, 2800]
                # Get diced feats
                feats = []
                for i in range((self.config.TIME_RANGE[1] - self.config.TIME_RANGE[0]) // self.config.SLICE_LENGTH):
                    feats.append(cwtmatr[:, self.config.SLICE_LENGTH * i: self.config.SLICE_LENGTH * (i + 1)])
                feats = np.array(feats)
                tf_feats.append(feats)

                # Get IMFs labels
                IMFs.append(IMFs_all.item().get('s{}'.format(channel)))
            tf_feats = np.array(tf_feats)
            tf_feats = np.concatenate(tf_feats, axis=0). \
                reshape([self.config.SLICE_LENGTH, self.config.SLICE_LENGTH, -1])  # [144, 144, 57]
            IMFs = np.array(IMFs)
            IMFs = np.concatenate(IMFs, axis=0). \
                reshape([self.config.TIME_RANGE[1] - self.config.TIME_RANGE[0], -1])  # [2800, 9]

            # Get Loc&Depth
            loc_dep = np.load(os.path.join(self.config.LABEL_ROOT_DIR, nth + '.npy'), allow_pickle=True)
            loc = loc_dep.item().get('loc')
            depth = loc_dep.item().get('depth')
            if np.max(loc) == 0:
                damage_matrix = np.array([0., 1.]).reshape([2])
            else:
                damage_matrix = np.array([1., 0.]).reshape([2])

            # [(144, 144, 57), (2800, 9), (110, 80), [2, 1], 1]
            data_pairs.append([tf_feats, IMFs, loc, damage_matrix, depth])
            final['feats'] = tf_feats
            final['imfs'] = IMFs
            final['loc'] = loc
            final['damage_matrix'] = damage_matrix
            final['depth'] = depth
            final_path = os.path.join(self.config.FINAL_ROOT_DIR, nth+'.npy')
            np.save(final_path, final)

        return data_pairs

    def get_final(self):
        final_names = os.listdir(self.config.FINAL_ROOT_DIR)
        data_pairs = []
        for fn in final_names:
            if fn == '.DS_Store':
                continue
            final = np.load(os.path.join(self.config.FINAL_ROOT_DIR, fn), allow_pickle=True)
            feats = final.item().get('feats')
            imfs = final.item().get('imfs') / self.config.SCALE
            loc = final.item().get('loc')
            damage_matrix = final.item().get('damage_matrix')
            depth = final.item().get('depth')
            depth = np.array(depth).reshape([1, ])
            data_pairs.append([feats, imfs, loc, damage_matrix, depth])
        return data_pairs

    def generator(self):
        data_pairs = self.get_final()
        n = len(data_pairs)
        for i in itertools.count(1):
            idx = i % n
            yield data_pairs[idx][0], data_pairs[idx][1], \
                  data_pairs[idx][2], data_pairs[idx][3], data_pairs[idx][4]


if __name__ == '__main__':
    from config import Config

    cfg = Config()

    # data_generator = DataGenerator(cfg)
    # data_pairs = data_generator.get_final()
    # for d in data_pairs:
    #     a = np.array(d[-1]).reshape([1,])
    #     print(a.shape)

    data_generator = DataGenerator(cfg).generator
    print(">>>>>>>>>>>>>>Dataset Initialization Done")
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             (tf.TensorShape([144, 144, 57]),
                                              tf.TensorShape([2800, 9]),
                                              tf.TensorShape([144, 72]),
                                              tf.TensorShape([2]),
                                              tf.TensorShape([None])))
    dataset = dataset.batch(8).shuffle(16)
    for feats, imfs, loc, damage_matrix, depth in dataset.take(2):
        print("{}--{}--{}--{}--{}".format(feats.shape, imfs.shape, loc.shape, damage_matrix.shape, depth.shape))
