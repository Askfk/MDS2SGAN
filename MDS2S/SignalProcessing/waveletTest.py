import pywt
import scipy.io as scio
import os
import matplotlib.pyplot as plt


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


mode = pywt.Modes.smooth


def plot_signal_decomp(data, w, title, level=7):
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

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))  # 重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i == 3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure(figsize=(16, 9))
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))


def show_and_pick():
    pass


file_names = os.listdir(ROOT_DIR)

for i in range(5):
    fn = file_names[i]

    data_path = os.path.join(ROOT_DIR, fn)
    data = scio.loadmat(data_path)['exportData']
    s1_1 = data[0][0][0][0]

    plot_signal_decomp(s1_1, 'coif5', fn, level=9)
# plot_signal_decomp(data2, 'sym5',
#          "DWT: Frequency and phase change - Symmlets5")
# plot_signal_decomp(s1_1, 'sym5', "DWT: Ecg sample - Symmlets5")

    plt.show()