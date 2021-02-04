"""Empirical modal decomposition."""

from scipy.signal import argrelextrema
from scipy import interpolate as spi
import numpy as np
import matplotlib.pyplot as plt


def sifting(data):
    index = list(range(len(data)))

    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值

    iy3_mean = (iy3_max + iy3_min) / 2
    return data - iy3_mean


def hasPeaks(data):
    max_peaks = list(set(argrelextrema(data, np.greater)[0]))
    min_peaks = list(set(argrelextrema(data, np.less)[0]))

    if len(max_peaks) > 3 and len(min_peaks) > 3:
        return True
    else:
        return False


# 判断IMFs
def isIMFs(data):
    max_peaks = list(set(argrelextrema(data, np.greater)[0]))
    print("Done max: {}".format(len(max_peaks)))
    min_peaks = list(set(argrelextrema(data, np.less)[0]))
    print("Done min: {}".format(len(min_peaks)))
    print(data[max_peaks].shape)
    if min(data[max_peaks]) < 0 or max(data[min_peaks]) > 0:
        return False
    else:
        return True


def getIMFs(data):
    while not isIMFs(data):
        data = sifting(data)
        print("LOOPING....")
    print("Done getIMF")
    return data


# EMD function
def EMD(data, num=8):
    IMFs = []
    i = 0
    while hasPeaks(data) or i < num:
        data_imf = getIMFs(data)
        data = data - data_imf
        IMFs.append(data_imf)
        i += 1
    return IMFs


def autoEMD(data, figsize=(18, 25), num=8):
    l = len(data)
    for j in range(l):
        IMFs = EMD(data[j], num=num)
        n = len(IMFs) + 1

        # 原始信号
        plt.figure(figsize=figsize)
        plt.subplot(n + 1, 1, 1)
        plt.plot(data[j], label='Origin')
        plt.title("Origin_{}".format(j + 1))

        final = 0
        # 若干条IMFs曲线
        for i in range(0, len(IMFs)):
            IMFs[i] = np.array(IMFs[i])
            temp = np.abs(IMFs[i])
            m = np.mean(temp)
            idx = np.where(temp >= 1.5 * m)
            IMFs[i][idx] = 0
            final += np.array(IMFs[i])
            plt.subplot(n + 1, 1, i + 3)
            plt.plot(IMFs[i])
            plt.ylabel('Amplitude')
            plt.title("IMFs " + str(i + 1))
        plt.subplot(n + 1, 1, 2)
        plt.plot(final)
        plt.ylabel('Amplitude')
        plt.title("IMFs " + str(n))

        plt.subplot(n + 1, 1, 2)
        plt.plot(final - np.array(data[j]))
        plt.ylabel('Residues')
        plt.title('Residues')

        # plt.legend()
        plt.xlabel('time (s)')
        plt.show()


if __name__ == '__main__':
    import scipy.io as scio
    import os

    ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'

    file_names = os.listdir(ROOT_DIR)
    print(file_names[0])
    fn = os.path.join(ROOT_DIR, file_names[0])

    data = scio.loadmat(fn)

    data = list(data.values())[6:12]
    imf = sifting(data[0])
    print("Done sifting: {}".format(imf.shape))
    imf = isIMFs(imf)
    print(imf)

