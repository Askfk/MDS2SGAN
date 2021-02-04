import os
import functools
import scipy.io as scio
import scipy.signal as scignal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial, log, floor
from sklearn.neighbors import KDTree
import pywt
from scipy.signal import argrelextrema
from scipy import interpolate as spi

from PyEMD import EMD, EEMD, Visualisation
from ..config import Config as cfg

DATA_ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'
LABEL_ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/labels'
RESULT_ROOT = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/EDAResults'


def custom_sort(x, y):
    x = x.split('-')
    y = y.split('-')
    if int(x[0]) < int(y[0]):
        return -1
    elif int(x[0]) > int(y[0]):
        return 1
    return 0


def get_damage():
    data_file_names = os.listdir(DATA_ROOT_DIR)
    data_file_names = list(map(lambda x: x[:-4], data_file_names))
    label_file_names = os.listdir(LABEL_ROOT_DIR)
    label_file_names.remove('.DS_Store')
    label_file_names = list(map(lambda x: x[:-4], label_file_names))
    data_file_names = list(sorted(data_file_names, key=functools.cmp_to_key(custom_sort)))
    label_file_names = list(sorted(label_file_names, key=functools.cmp_to_key(custom_sort)))

    damage = {}

    for l_file_name in label_file_names:
        fn = l_file_name.split('-')
        if fn[-1] not in damage.keys():
            damage[fn[-1]] = [l_file_name]
        else:
            mid = damage[fn[-1]]
            mid.append(l_file_name)
            damage[fn[-1]] = mid
    return damage


# TODO: 时域分析N步走！
# 1. 第一步分析类别数量，各个类别样本数量，然后得出数据采集的依据，例如增大有损无损recall，故而0类别（无损）数量相对较多
# 2. 第二步分析各个类别的主要分布位置，利用label中的坐标位置可视化热力图
# 3. 第三步参考kaggle，分析signal mean（信号均值），得出一些结论，需要编，参考kaggle
# 4. 第四步参考kaggle，分析Permutation entropy（排列熵），分析每个类别s0-s5之间的排列熵，得出不同通道相关性关系，然后从中挑选相关性较强的几个通道作为模型输入。
# 5. 第五步参考kaggle，分析Approximate entropy
# 6. 第六步参考kaggle，分析Higuchi fractal dimension（Higuchi分型维数）
# 7. 第七步参考kaggle，分析Katz fractal dimension（Katz分型维数）
class TimeAnalysis:

    def __init__(self, damage):
        self.damage = damage
        self.signals = self.extract_signals()

    def extract_signals(self, points=4000, scale=1):
        start = 2000
        signals = []
        for k in self.damage.keys():
            temp = []
            if len(k.split('-')[-1]) < 3:
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, damage[k][0] + '.mat'))
            else:
                params = self.damage[k][0].split('-')
                name = ''
                for p in params[:-1]:
                    name += p + '-'
                name += 'n'
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, name + '.mat'))
            temp.append(data['s0'][start: points, 0] / scale)
            temp.append(data['s1'][start: points, 0] / scale)
            temp.append(data['s2'][start: points, 0] / scale)
            temp.append(data['s3'][start: points, 0] / scale)
            temp.append(data['s4'][start: points, 0] / scale)
            temp.append(data['s5'][start: points, 0] / scale)
            signals.append(temp)

        return signals

    def class_num_analysis(self, show_results=False, save=False):
        """
        第一步分析类别数量，各个类别样本数量，然后得出数据采集的依据，例如增大有损无损recall，故而0类别（无损）数量相对较多
        :param show_results:
        :return:
        """
        classes = np.arange(len(self.damage.keys()))
        nums = list(map(lambda x: len(x), list(damage.values())))
        plt.rcParams['font.family'] = ['Times New Roman']
        fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=200)
        x = np.arange(len(classes))  # the label locations
        width = 0.35  # the width of the bars
        label_font = {
            'weight': 'bold',
            'size': 14,
            'family': 'simsun'
        }
        rects1 = ax.bar(x - width / 2, nums, width, label='2006', ec='k', color='gray', lw=.8)
        ax.tick_params(which='major', direction='in', length=5, width=1.5, labelsize=11, bottom=False)
        ax.tick_params(axis='x', labelsize=11, bottom=False, labelrotation=15)
        ax.set_xticks(x)

        ax.set_ylabel('(数量)', fontdict=label_font)
        ax.set_xticklabels(classes, fontdict=label_font)

        self.__autolabel(rects1, ax)
        fig.tight_layout()
        plt.title('Class_Num_Analysis')
        if save:
            plt.savefig(os.path.join(RESULT_ROOT, 'class_num_analysis.png'), dpi=600, bbox_inches='tight')

        if show_results:
            plt.show()
        return classes, nums

    def show_position(self, show_results=False, save=False):
        """
        第二步分析各个类别的主要分布位置，利用label中的坐标位置可视化热力图
        :param show_results:
        :return:
        """
        overall_pos = {}
        i = 0
        plt.figure(figsize=(40, 30))
        for key in self.damage.keys():
            if len(key.split('-')[-1]) < 3:
                v = damage[key][0]
                position_path = os.path.join(LABEL_ROOT_DIR, v + '.txt')
                pos = np.loadtxt(position_path)
                idx = np.where(pos == 1)
                if idx[0].shape[0] is not 0:
                    title = str(idx[0][0]) + " " + str(idx[1][0]) + " " + str(idx[0][-1]) + " " + str(idx[1][0]) + " " \
                            + str(idx[0][-1]) + " " + str(idx[1][-1]) + " " + str(idx[0][-1]) + " " + str(idx[1][-1])
                else:
                    title = str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " \
                            + str(0) + " " + str(0) + " " + str(0) + " " + str(0)
            else:
                pos = list(map(int, key.split('-')[-1].split()))
                pos = self.__generate_heatmap(pos)
                title = key
            save_path = os.path.join(RESULT_ROOT, str(i) + '_position_heatmap.png')
            plt.subplot(5, 6, i+1)
            sns.heatmap(pos, annot=False)
            plt.title(str(i))
            if save:
                plt.savefig(save_path, dpi=600, bbox_inches='tight')
            overall_pos[key] = save_path
            i += 1
        if show_results:
            plt.show()
        return overall_pos

    def signal_mean(self, show_results=False, save=False, scale=2000):
        """
        第三步参考kaggle，分析signal mean（信号均值），得出一些结论，需要编，参考kaggle
        :param save:
        :param show_results:
        :param scale:
        :return:
        """
        x = list(range(6))
        i = 0
        plt.figure(figsize=(40, 30))
        for temp in self.signals:
            plt.subplot(5, 6, i + 1)
            sns.boxplot(x=x, y=temp)
            sns.pointplot(x=x, y=[np.median(s) for s in temp], color='black', linestyles=['-'], scale=0.5)
            plt.title(str(i))
            if save:
                plt.savefig(os.path.join(RESULT_ROOT, str(i) + "_signal mean.png"), dpi=600, bbox_inches='tight')
            i += 1
        if show_results:
            plt.show()

    def permutation_entropy(self, show_results=False, save=False):
        """
        第四步参考kaggle，分析Permutation entropy（排列熵），分析每个类别s0-s5之间的排列熵，得出不同通道相关性关系，
        然后从中挑选相关性较强的几个通道作为模型输入。
        """
        perm_entropy = []
        x = list(range(6))
        i = 0
        for signal in self.signals:
            temp = []
            for s in signal:
                temp.append(self._perm_entropy(s))
            perm_entropy.append(temp)
        perm_entropy = np.array(perm_entropy)
        perm_entropy = perm_entropy.reshape([6, -1])
        print(perm_entropy.shape)
        sns.boxplot(x=x, y=perm_entropy.tolist())
        sns.pointplot(x=x, y=[np.median(s) for s in perm_entropy], color='black', linestyles=['-'], scale=0.5)
        plt.title("permutation_entropy")
        if save:
            plt.savefig(os.path.join(RESULT_ROOT, "perm_entropy.png"), dpi=600, bbox_inches='tight')
        if show_results:
            plt.show()

        return perm_entropy

    def approximate_entropy(self, order=2, metric='chebyshev', show_results=False, save=False):
        """
        第五步参考kaggle，分析Approximate entropy
        :param x:
        :param order:
        :param metric:
        :return:
        """
        app_entropy = []
        x = list(range(6))
        i = 0
        for signal in self.signals:
            temp = []
            for s in signal:
                phi = self._app_samp_entropy(s, order=order, metric=metric, approximate=True)
                temp.append(phi)
            app_entropy.append(temp)

        app_entropy = np.array(app_entropy)
        app_entropy = app_entropy.reshape([6, -1])
        print(app_entropy.shape)
        sns.boxplot(x=x, y=app_entropy.tolist())
        sns.pointplot(x=x, y=[np.median(s) for s in app_entropy], color='black', linestyles=['-'], scale=0.5)
        plt.title("app_entropy")
        if save:
            plt.savefig(os.path.join(RESULT_ROOT, "app_entropy.png"), dpi=600, bbox_inches='tight')
        i += 1
        if show_results:
            plt.show()
        return app_entropy

    def higuchi_fractal_dimension(self, kmax=10, show_results=False, save=False):
        """
        第六步参考kaggle，分析Higuchi fractal dimension（Higuchi分型维数）
        :param x:
        :param kmax:
        :return:
        """
        higuchi_fd = []
        x = list(range(6))
        i = 0
        for signal in self.signals:
            temp = []
            for s in signal:
                s = np.asarray(s, dtype=np.float64)
                kmax = int(kmax)
                temp.append(self._higuchi_fd(s, kmax))
            higuchi_fd.append(temp)
        higuchi_fd = np.array(higuchi_fd)
        higuchi_fd = higuchi_fd.reshape([6, -1])
        print(higuchi_fd.shape)
        sns.boxplot(x=x, y=higuchi_fd.tolist())
        sns.pointplot(x=x, y=[np.median(s) for s in higuchi_fd], color='black', linestyles=['-'], scale=0.5)
        plt.title("higuchi_fd")
        if save:
            plt.savefig(os.path.join(RESULT_ROOT, "higuchi_fd.png"), dpi=600, bbox_inches='tight')
        i += 1
        if show_results:
            plt.show()
        return higuchi_fd

    def katz_fractal_dimension(self, show_results=False, save=False):
        """
        第七步参考kaggle，分析Katz fractal dimension（Katz分型维数）
        :param x:
        :return:
        """
        katz_fd = []
        x = list(range(6))
        i = 0
        for signal in self.signals:
            temp = []
            for s in signal:
                s = np.array(s)
                dists = np.abs(np.ediff1d(s))
                ll = dists.sum()
                ln = np.log10(np.divide(ll, dists.mean()))
                aux_d = s - s[0]
                d = np.max(np.abs(aux_d[1:]))
                temp.append(np.divide(ln, np.add(ln, np.log10(np.divide(d, ll)))))
            katz_fd.append(temp)

        katz_fd = np.array(katz_fd)
        katz_fd = katz_fd.reshape([6, -1])
        print(katz_fd.shape)
        sns.boxplot(x=x, y=katz_fd.tolist())
        sns.pointplot(x=x, y=[np.median(s) for s in katz_fd], color='black', linestyles=['-'], scale=0.5)
        plt.title("katz_fd")
        if save:
            plt.savefig(os.path.join(RESULT_ROOT, "katz_fd.png"), dpi=600, bbox_inches='tight')
        i += 1
        if show_results:
            plt.show()
        return katz_fd

    def _perm_entropy(self, x, order=3, delay=1, normalize=True):
        x = np.array(x)
        ran_order = range(order)
        hashmult = np.power(order, ran_order)
        # Embed x and sort the order of permutations
        sorted_idx = self._embed(x, order=order, delay=delay).argsort(kind='quicksort')
        # Associate unique integer to each permutations
        hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
        # Return the counts
        _, c = np.unique(hashval, return_counts=True)
        # Use np.true_divide for Python 2 compatibility
        p = np.true_divide(c, c.sum())
        pe = -np.multiply(p, np.log2(p)).sum()
        if normalize:
            pe /= np.log2(factorial(order))
        return pe

    def __autolabel(self, rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def __generate_heatmap(self, params):
        [x1, y1] = [params[0], params[1]]
        [x2, y2] = [params[2], params[3]]
        [x3, y3] = [params[6], params[7]]
        [x4, y4] = [params[4], params[5]]

        upk = (y2 - y1) / (x2 - x1)
        upb = y1 - upk * x1
        rightk = (y3 - y2) / (x3 - x2)
        rightb = y2 - rightk * x2
        downk = (y4 - y3) / (x4 - x3)
        downb = y3 - downk * x3
        leftk = (y1 - y4) / (x1 - x4)
        leftb = y4 - leftk * x4
        sx, ex = x1, x3
        sy, ey = y2, y4
        label = np.zeros([cfg.LOCAL_HEIGHT, cfg.LOCAL_WIDTH])
        for tx in range(sx, ex):
            for ty in range(sy, ey):
                if ty >= upk * tx + upb:
                    if ty >= rightk * tx + rightb:
                        if ty <= downk * tx + downb:
                            if ty <= leftk * tx + leftb:
                                label[ty, tx] = 1
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
        return label

    def _embed(self, x, order=3, delay=1):
        N = len(x)
        if order * delay > N:
            raise ValueError("Error: order * delay should be lower than x.size")
        if delay < 1:
            raise ValueError("Delay has to be at least 1.")
        if order < 2:
            raise ValueError("Order has to be at least 2.")
        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[i * delay:i * delay + Y.shape[1]]
        return Y.T

    def _app_samp_entropy(self, x, order, metric='chebyshev', approximate=True):
        """Utility function for `app_entropy`` and `sample_entropy`.
        """
        _all_metrics = KDTree.valid_metrics
        if metric not in _all_metrics:
            raise ValueError('The given metric (%s) is not valid. The valid '
                             'metric names are: %s' % (metric, _all_metrics))
        phi = np.zeros(2)
        r = 0.2 * np.std(x, axis=-1, ddof=1)

        # compute phi(order, r)
        _emb_data1 = self._embed(x, order, 1)
        if approximate:
            emb_data1 = _emb_data1
        else:
            emb_data1 = _emb_data1[:-1]
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        # compute phi(order + 1, r)
        emb_data2 = self._embed(x, order + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                               count_only=True
                                                               ).astype(np.float64)
        if approximate:
            phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
            phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
        else:
            phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
            phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
        return phi

    def _numba_sampen(self, x, mm=2, r=0.2):
        """
        Fast evaluation of the sample entropy using Numba.
        """
        n = x.size
        n1 = n - 1
        mm += 1
        mm_dbld = 2 * mm

        # Define threshold
        r *= x.std()

        # initialize the lists
        run = [0] * n
        run1 = run[:]
        r1 = [0] * (n * mm_dbld)
        a = [0] * mm
        b = a[:]
        p = a[:]

        for i in range(n1):
            nj = n1 - i

            for jj in range(nj):
                j = jj + i + 1
                if abs(x[j] - x[i]) < r:
                    run[jj] = run1[jj] + 1
                    m1 = mm if mm < run[jj] else run[jj]
                    for m in range(m1):
                        a[m] += 1
                        if j < n1:
                            b[m] += 1
                else:
                    run[jj] = 0
            for j in range(mm_dbld):
                run1[j] = run[j]
                r1[i + n * j] = run[j]
            if nj > mm_dbld - 1:
                for j in range(mm_dbld, nj):
                    run1[j] = run[j]

        m = mm - 1

        while m > 0:
            b[m] = b[m - 1]
            m -= 1

        b[0] = n * n1 / 2
        a = np.array([float(aa) for aa in a])
        b = np.array([float(bb) for bb in b])
        p = np.true_divide(a, b)
        return -log(p[-1])

    def _log_n(self, min_n, max_n, factor):
        max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
        ns = [min_n]
        for i in range(max_i + 1):
            n = int(floor(min_n * (factor ** i)))
            if n > ns[-1]:
                ns.append(n)
        return np.array(ns, dtype=np.int64)

    def _higuchi_fd(self, x, kmax):
        n_times = x.size
        lk = np.empty(kmax)
        x_reg = np.empty(kmax)
        y_reg = np.empty(kmax)
        for k in range(1, kmax + 1):
            lm = np.empty((k,))
            for m in range(k):
                ll = 0
                n_max = floor((n_times - m - 1) / k)
                n_max = int(n_max)
                for j in range(1, n_max):
                    ll += abs(x[m + j * k] - x[m + (j - 1) * k])
                ll /= k
                ll *= (n_times - 1) / (k * n_max)
                lm[m] = ll
            # Mean of lm
            m_lm = 0
            for m in range(k):
                m_lm += lm[m]
            m_lm /= k
            lk[k - 1] = m_lm
            x_reg[k - 1] = log(1. / k)
            y_reg[k - 1] = log(m_lm)
        higuchi, _ = self._linear_regression(x_reg, y_reg)
        return higuchi

    def _linear_regression(self, x, y):
        n_times = x.size
        sx2 = 0
        sx = 0
        sy = 0
        sxy = 0
        for j in range(n_times):
            sx2 += x[j] ** 2
            sx += x[j]
            sxy += x[j] * y[j]
            sy += y[j]
        den = n_times * sx2 - (sx ** 2)
        num = n_times * sxy - sx * sy
        slope = num / den
        intercept = np.mean(y) - slope * np.mean(x)
        return slope, intercept


# TODO: 频域分析N步走
# 1. 第一步，
class FrequencyAnalysis:
    def __init__(self, damage):
        self.damage = damage
        self.signals = self.extract_signals()

    def extract_signals(self, scale=1, points=None):
        signals = []
        points = points or 10000
        for k in self.damage.keys():
            temp = []
            if len(k.split('-')[-1]) < 3:
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, damage[k][0] + '.mat'))
            else:
                params = self.damage[k][0].split('-')
                name = ''
                for p in params[:-1]:
                    name += p + '-'
                name += 'n'
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, name + '.mat'))
            temp.append(data['s0'][: points, 0] / scale)
            temp.append(data['s1'][: points, 0] / scale)
            temp.append(data['s2'][: points, 0] / scale)
            temp.append(data['s3'][: points, 0] / scale)
            temp.append(data['s4'][: points, 0] / scale)
            temp.append(data['s5'][: points, 0] / scale)
            signals.append(temp)

        return signals

    def FFT(self, points=None, show_results=False, save=False, figsize=(36, 25), show_range=None):
        points = points or len(self.signals[0][0])
        t = np.arange(0, points) * 24000 / points
        j = 0
        fft_results = []
        show_range = show_range or (0, points)
        for temp in self.signals:
            shape = (3, len(temp) // 3)
            fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize)
            fft_temp = []
            for i in range(len(temp)):
                fft = np.power(np.abs(np.power(np.fft.fft(temp[i], points), 2)), 0.5)
                fft_temp.append(fft)
                axs[i // 2, i % 2].plot(t[show_range[0]: show_range[1]], fft[show_range[0]: show_range[1]])
                axs[i // 2, i % 2].set_title('s{} energy spectrum'.format(j, i))
            fig.suptitle('class {}'.format(j))
            if save:
                plt.savefig(os.path.join(RESULT_ROOT, str(i) + "_class_fft_energy_spectrum.png"), dpi=600,
                            bbox_inches='tight')
            if show_results:
                plt.show()
            j += 1
            fft_results.append(fft_temp)

        return fft_results

    def STFT(self, show_results=False, save=False, figsize=(25, 16), show_range=(0, 1500), nperseg=128):
        i = 0
        stft_results = []
        for temp in self.signals:
            shape = (3, len(temp) // 3)
            stft_temp = []
            plt.figure(figsize=figsize)
            for j in range(len(temp)):
                params = scignal.stft(temp[j], fs=24e3, nperseg=nperseg)
                f, t, zxx = params
                plt.subplot(shape[0], shape[1], j+1)
                plt.pcolormesh(t, f, np.abs(zxx))
                plt.colorbar()
                plt.title('S{} STFT Magnitude'.format(j))
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.ylim(show_range[0], show_range[1])
                stft_temp.append(params)
            # plt.suptitle('Class {}'.format(i))
            stft_results.append(stft_temp)
            if save:
                plt.savefig(os.path.join(RESULT_ROOT, str(i) + "_class_sfft_spectrum.png"), dpi=600,
                            bbox_inches='tight')
            i += 1
            if show_results:
                plt.show()
        return stft_results

    def wavelet(self, show_results=False, save=False, w='cgau8', sampling_rate=24e3, figsize=(36, 25), points=None):
        """

        :param figsize:
        :param sampling_rate: 采样频率
        :param show_results:
        :param save:
        :param w: 小波函数种类
        :return:
        """
        wavelet_results = []
        totalscal = len(self.signals[0][0])
        points = points or totalscal
        t = np.arange(0, points) * 24000 / points
        fc = pywt.central_frequency(w)  # 中心频率
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        i = 0
        for temp in self.signals:
            shape = (3, len(temp) // 3)
            # fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize)
            wavelet_temp = []
            for j in range(len(temp)):
                [cwtmatr, frequencies] = pywt.cwt(temp[j], scales, w, 1.0 / sampling_rate)  # 连续小波变换
                plt.subplot(shape[0], shape[1], j+1)
                plt.contourf(t, frequencies, abs(cwtmatr))
                plt.ylabel(u"freq(Hz)")
                plt.xlabel(u"time(s)")
                plt.title('S{} Wavelet Magnitude'.format(j))
                wavelet_temp.append([cwtmatr, frequencies])
            wavelet_results.append(wavelet_temp)

            if save:
                plt.savefig(os.path.join(RESULT_ROOT, str(i) + "_class_wavelet_spectrum.png"), dpi=600,
                            bbox_inches='tight')
            i += 1
            if show_results:
                plt.show()
        return wavelet_results


class ModeDecomposition:

    def __init__(self, damage):
        self.damage = damage
        self.signals = self.extract_signals()

    def extract_signals(self, points=10000, scale=1):
        signals = []
        for k in self.damage.keys():
            temp = []
            if len(k.split('-')[-1]) < 3:
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, damage[k][0] + '.mat'))
            else:
                params = self.damage[k][0].split('-')
                name = ''
                for p in params[:-1]:
                    name += p + '-'
                name += 'n'
                data = scio.loadmat(os.path.join(DATA_ROOT_DIR, name + '.mat'))
            temp.append(data['s0'][: points, 0] / scale)
            temp.append(data['s1'][: points, 0] / scale)
            temp.append(data['s2'][: points, 0] / scale)
            temp.append(data['s3'][: points, 0] / scale)
            temp.append(data['s4'][: points, 0] / scale)
            temp.append(data['s5'][: points, 0] / scale)
            signals.append(temp)

        return signals

    def EMD(self, show_results=False, save=False, figsize=(18, 25), num=8):
        n = np.random.randint(0, len(self.signals))
        data = self.signals[n]
        l = len(data)
        for j in range(l):
            IMFs = self._EMD(data[j], num=num)
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

            if save:
                plt.savefig(os.path.join(RESULT_ROOT, "S{}_EMD_decomposition.png".format(j)), dpi=600,
                            bbox_inches='tight')
            if show_results:
                plt.show()

    def VMD(self, show_results=False, save=False, alpha=500, tau=0., K=3, DC=0, init=1, tol=1e-7):
        n = np.random.randint(0, len(self.signals))
        data = self.signals[n]
        m = len(data)
        for j in range(m):
            signal = data[j]

            f_hat = np.fft.fftshift((np.fft.fft(signal)))
            u, u_hat, omega = self._VMD(signal, alpha, tau, K, DC, init, tol)
            t = np.arange(0, 8000)

            plt.figure(figsize=(16, 16))

            n = len(u)
            plt.subplot(n + 2, 1, 1)
            plt.plot(signal)
            plt.title('original')

            combined = 0

            for i in range(n):
                plt.subplot(n + 2, 1, i + 3)
                plt.plot(u[i])
                combined += u[i]

                plt.title('Decomposed modes_{}'.format(i + 1))

            plt.subplot(n + 2, 1, 2)
            plt.plot(combined)
            plt.plot(combined - np.array(signal))
            plt.title('combined')
            if save:
                plt.savefig(os.path.join(RESULT_ROOT, "S{}_VMD_decomposition.png".format(j)), dpi=600,
                            bbox_inches='tight')
            if show_results:
                plt.show()

    def Wavelet(self, show_results=False, save=False, mode=pywt.Modes.smooth, w='coif5', level=7):
        """

        :param level: 分解等级数
        :param show_results:
        :param save:
        :param mode:
        :param w: 小波函数种类
        :return:
        """
        w = pywt.Wavelet(w)  # 选取小波函数
        ca = []  # 近似分量
        cd = []  # 细节分量
        n = np.random.randint(0, len(self.signals))
        sample = self.signals[n]
        num = 0
        for s in sample:
            a = s
            for i in range(level):
                (a, d) = pywt.dwt(a, w, mode)  # 进行level阶离散小波变换
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
            ax_main.set_title("S{}".format(num))
            ax_main.plot(s)
            ax_main.set_xlim(0, len(s) - 1)

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

            if save:
                plt.savefig(os.path.join(RESULT_ROOT, "S{}_wavelet_decomposition.png".format(num)), dpi=600,
                            bbox_inches='tight')
            num += 1
            if show_results:
                plt.show()

    def GDMD(self):
        pass

    def sifting(self, data):
        index = list(range(len(data)))

        max_peaks = list(argrelextrema(data, np.greater)[0])
        min_peaks = list(argrelextrema(data, np.less)[0])

        ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # 样本点导入，生成参数
        iy3_max = spi.splev(index, ipo3_max)  # 根据观测点和样条参数，生成插值

        ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # 样本点导入，生成参数
        iy3_min = spi.splev(index, ipo3_min)  # 根据观测点和样条参数，生成插值

        iy3_mean = (iy3_max + iy3_min) / 2
        return data - iy3_mean

    def hasPeaks(self, data):
        max_peaks = list(set(argrelextrema(data, np.greater)[0]))
        min_peaks = list(set(argrelextrema(data, np.less)[0]))

        if len(max_peaks) > 3 and len(min_peaks) > 3:
            return True
        else:
            return False

    # 判断IMFs
    def isIMFs(self, data):
        max_peaks = list(set(argrelextrema(data, np.greater)[0]))
        # print("Done max: {}".format(len(max_peaks)))
        min_peaks = list(set(argrelextrema(data, np.less)[0]))
        # print("Done min: {}".format(len(min_peaks)))
        # print(data[max_peaks].shape)
        if min(data[max_peaks]) < 0 or max(data[min_peaks]) > 0:
            return False
        else:
            return True

    def getIMFs(self, data):
        while not self.isIMFs(data):
            data = self.sifting(data)
            # print("LOOPING....")
        # print("Done getIMF")
        return data

    # EMD function
    def _EMD(self, data, num=8):
        IMFs = []
        i = 0
        while self.hasPeaks(data) or i < num:
            data_imf = self.getIMFs(data)
            data = data - data_imf
            IMFs.append(data_imf)
            i += 1
        return IMFs

    def _VMD(self, f, alpha, tau, K, DC, init, tol):
        """
        u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
        Variational mode decomposition
        Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
        code based on Dominique Zosso's MATLAB code, available at:
        https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
        Original paper:
        Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
        IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


        Input and Parameters:
        ---------------------
        f       - the time domain signal (1D) to be decomposed
        alpha   - the balancing parameter of the data-fidelity constraint
        tau     - time-step of the dual ascent ( pick 0 for noise-slack )
        K       - the number of modes to be recovered
        DC      - true if the first mode is put and kept at DC (0-freq)
        init    - 0 = all omegas start at 0
                  1 = all omegas start uniformly distributed
                  2 = all omegas initialized randomly
        tol     - tolerance of convergence criterion; typically around 1e-6
        Output:
        -------
        u       - the collection of decomposed modes
        u_hat   - spectra of the modes
        omega   - estimated mode center-frequencies
        """

        if len(f) % 2:
            f = f[:-1]

        # Period and sampling frequency of input signal
        fs = 1. / len(f)

        ltemp = len(f) // 2
        fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
        fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

        # Time Domain 0 to T (of mirrored signal)
        T = len(fMirr)
        t = np.arange(1, T + 1) / T

        # Spectral Domain discretization
        freqs = t - 0.5 - (1 / T)

        # Maximum number of iterations (if not converged yet, then it won't anyway)
        Niter = 500
        # For future generalizations: individual alpha for each mode
        Alpha = alpha * np.ones(K)

        # Construct and center f_hat
        f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
        f_hat_plus = np.copy(f_hat)  # copy f_hat
        f_hat_plus[:T // 2] = 0

        # Initialization of omega_k
        omega_plus = np.zeros([Niter, K])

        if init == 1:
            for i in range(K):
                omega_plus[0, i] = (0.5 / K) * (i)
        elif init == 2:
            omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
        else:
            omega_plus[0, :] = 0

        # if DC mode imposed, set its omega to 0
        if DC:
            omega_plus[0, 0] = 0

        # start with empty dual variables
        lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

        # other inits
        uDiff = tol + np.spacing(1)  # update step
        n = 0  # loop counter
        sum_uk = 0  # accumulator
        # matrix keeping track of every iterant // could be discarded for mem
        u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

        # *** Main loop for iterative updates***

        while uDiff > tol and n < Niter - 1:  # not converged and below iterations limit
            # update first mode accumulator
            k = 0
            sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

            # update spectrum of first mode through Wiener filter of residuals
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1. + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)

            # update first omega if not held at 0
            if not (DC):
                omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                    abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

            # update of any other mode
            for k in np.arange(1, K):
                # accumulator
                sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
                # mode spectrum
                u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                        1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
                # center frequencies
                omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                    abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

            # Dual ascent
            lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

            # loop counter
            n = n + 1

            # converged yet?
            uDiff = np.spacing(1)
            for i in range(K):
                uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                                 np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))

            uDiff = np.abs(uDiff)

            # Postprocessing and cleanup

        # discard empty space if converged early
        Niter = np.min([Niter, n])
        omega = omega_plus[:Niter, :]

        idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
        # Signal reconstruction
        u_hat = np.zeros([T, K], dtype=complex)
        u_hat[T // 2:T, :] = u_hat_plus[Niter - 1, T // 2:T, :]
        u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2:T, :])
        u_hat[0, :] = np.conj(u_hat[-1, :])

        u = np.zeros([K, len(t)])
        for k in range(K):
            u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

        # remove mirror part
        u = u[:, T // 4:3 * T // 4]

        # recompute spectrum
        u_hat = np.zeros([u.shape[1], K], dtype=complex)
        for k in range(K):
            u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

        return u, u_hat, omega


damage = get_damage()
t = TimeAnalysis(damage)
# classes, nums = t.class_num_analysis(show_results=True)
# overall_pos = t.show_position(show_results=True, save=False)
t.signal_mean(show_results=True)  # NEED TO ABS THE SIGNALS, CURRENTLY CANT WORK
# t.permutation_entropy(show_results=True)
# t.approximate_entropy(show_results=True) # ABANDONED
# t.higuchi_fractal_dimension(show_results=True)
# t.katz_fractal_dimension(show_results=True) # ABANDONED


fa = FrequencyAnalysis(damage)
# fa.FFT(show_results=True, show_range=[0, 300])
# fa.STFT(show_results=True)
# fa.wavelet(show_results=True)  # CANT WORK


# md = ModeDecomposition(damage)
# md.EMD(show_results=True)  # CANT WORK
# md.VMD(show_results=True)
# md.Wavelet(show_results=True)