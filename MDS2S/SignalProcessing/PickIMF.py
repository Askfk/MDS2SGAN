import numpy as np
import os
import scipy.io as scio
import matplotlib.pyplot as plt
from new.EDA import WaveletDenoising
import pywt
mode = pywt.Modes.smooth

DATA_ROOT_DIR = '/Users/zcc/Documents/wield/data'
IMF_ROOT_DIR = '/Users/zcc/Documents/wield/IMFs'

data_file_names = os.listdir(DATA_ROOT_DIR)
n = len(data_file_names)

t = np.arange(0, 10000 / 24000, 1 / 24000)
f = np.arange(0, 10000) * 24000 / 10000
r = (700, 3500)
nth = [4, 5]


def plot_signal_decomp(t, data, w, title, level=9):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    """
    w = pywt.Wavelet(w)  # 选取小波函数
    a = data
    ca = []  # 近似分量
    cd = []  # 细节分量
    for i in range(level):
        (a, d) = pywt.dwt(a, w, mode)  # 进行level阶离散小波变换
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

    recd = list(map(lambda x: x[:nt], rec_d))
    return recd


for i in range(n):
    fn = os.path.join(DATA_ROOT_DIR, data_file_names[i])
    data = scio.loadmat(fn)
    s0 = data['s0'][:, 0]
    s1 = data['s1'][:, 0]
    s2 = data['s2'][:, 0]
    s3 = data['s3'][:, 0]
    s4 = data['s4'][:, 0]
    s5 = data['s5'][:, 0]

    data = [s0, s1, s2, s3, s4, s5]
    IMFs = {}
    save_path = os.path.join(IMF_ROOT_DIR, data_file_names[i].split('-')[0] + '.npy')

    for j in range(6):
        s = data[j]
        d = WaveletDenoising(s).out[r[0]: r[1]]

        rec_d = plot_signal_decomp(t[r[0]: r[1]], d, 'coif5', data_file_names[i]+"_S{}".format(j), level=6)
        # plt.show()

        # print("Please Choose the IMFs for S{}".format(j))
        # nth = input().split()
        # nth = list(map(lambda x: int(x) - 1, nth))
        imfs = []
        imf = 0
        for nt in nth:
            imfs.append(rec_d[nt])
            imf += rec_d[nt]
        imfs.append(d - imf)

        IMFs['s{}'.format(j)] = imfs

    np.save(save_path, IMFs)
    print("Done file {}".format(fn))