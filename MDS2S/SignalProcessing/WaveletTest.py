import pywt
import scipy.io as scio
import os
import matplotlib.pyplot as plt
from EDA import WaveletDenoising
import numpy as np
from TheRealEDA import FrequencyAnalysis, get_damage


ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'


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


mode = pywt.Modes.smooth


def plot_signal_decomp(t, data, w, title, level=9):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)  # 选取小波函数
    a = data
    ca = []  # 近似分量
    cd = []  # 细节分量
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)  # 进行5阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []
    nt = len(t)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))  # 重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    return rec_d

    fig = plt.figure(figsize=(9, 12))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(t, data)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(t, y[:nt], 'r')

        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(t, y[:nt], 'g')

        ax.set_ylabel("D%d" % (i + 1))
    recd = list(map(lambda x: x[:nt], rec_d))
    return recd


if __name__ == '__main__':
    file_names = os.listdir(ROOT_DIR)

    file_name = '165-24-10000-400-6500-20-n.mat'

    data_path = os.path.join(ROOT_DIR, file_name)

    data = scio.loadmat(data_path)
    s1 = data['s1'][:, 0]
    d = WaveletDenoising(s1).out[700:3500]
    t = np.arange(0, 10000 / 24000, 1 / 24000)


    rec_d = plot_signal_decomp(t[700:3500], d, 'coif5', 'Wavelet', level=6)

    plt.figure(figsize=(9, 6))
    for i in range(2):
        plt.subplot(3, 1, i+1)
        plt.plot(t[700:3500], rec_d[-(i+1)][:2800])
        # plt.title("模态分量 {}".format(i+1))
        d = d[:2800] - rec_d[-(i+1)][:2800]
    plt.subplot(3, 1, 3)
    plt.plot(t[700:3500], d[:2800])
    # plt.title("噪声模态分量")

    plt.show()

