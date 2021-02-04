"""
Exploratory Data Analysis
For this part, please refer to
https://www.kaggle.com/ymleeeee/ion-switching-competition-signal-eda/edit
"""

import os
import gc
import time
import math
from numba import jit
from math import log, floor, factorial
from sklearn.neighbors import KDTree

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pywt
from statsmodels.robust import mad

import scipy
from scipy import signal
from scipy.signal import butter, deconvolve


def average_smoothing(signal, kernel_size=3, stride=1):
    sample = [signal[0]]
    start = 0
    end = kernel_size
    while end <= len(signal):
        start = start + stride
        end = end + stride
        sample.append(np.mean(signal[start:end]))
        # sample.extend(np.ones(kernel_size) * np.mean(signal[start:end]))
    sample.append(signal[-1])
    return np.array(sample)


class WaveletDenoising:
    """
    Wavelet denoising is a way to remove the unnecessary noise from a signal. This method calculates coefficients
    called the "wavelet coefficients". These coefficients decide which pieces of information to keep (signal) and
    which ones to discard (noise).

    We make use of the MAD value (mean absolute deviation) to understand the randomness in the signal and accordingly
    decide the minimum threshold for the wavelet coefficients in the time series. We filter out the low coefficients
    from the wavelet coefficients and reconstruct the electric signal from the remaining coefficients and that's it;
    we have successfully removed noise from the electric signal.
    """
    def __init__(self, x):
        self.out = self.denoise_signal(x)

    def maddest(self, d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def denoise_signal(self, x, wavelet='db4', level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1 / 0.6745) * self.maddest(coeff[-level])

        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

        return pywt.waverec(coeff, wavelet, mode='per')


class WaveEntropy:
    """
    Entropy utilities.
    """

    all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy', 'sample_entropy']

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

    def perm_entropy(self, x, order=3, delay=1, normalize=False):
        """
        The permutation entropy is a complexity measure for time-series first introduced by Bandt and Pompe
        in 2002. It represents the information contained in comparing n consecutive values of the time series.
        It is a measure of entropy or disorderliness in a time series.
        """

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
        pe = -np.multiply(p, np.log2(p).sum())
        if normalize:
            pe /= np.log2(factorial(order))
        return pe

    def _app_samp_entropy(self, x, order, metric='chebyshev', approximate=True):
        """
        Utility function for 'app_entropy' and 'sample_entropy'.
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
        count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r, count_only=True).astype(np.float64)

        # compute phi(order + 1, r)
        emb_data2 = self._embed(x, order + 1, 1)
        count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r, count_only=True).astype(np.float64)

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

    def app_entropy(self, x, order=2, metric='chebyshev'):
        """
        Approximate entropy is a technique used to quantify the amount of regularity and the
        unpredictability of fluctuations over time-series data. Smaller values indicates that
        the data is more regular and predictable.
        """

        phi = self._app_samp_entropy(x, order=order, metric=metric, approximate=True)
        return np.subtract(phi[0], phi[1])


class HiguchiFractalDimension:
    """
    The Higuchi fractal dimension is a method to calculate the fractal dimension of any
    two-dimensional curve. Generally, curves with higher fractal dimension are "rougher"
    or more "complex" (higher entropy).
    """

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

    def higuchi_fd(self, x, kmax=10):
        x = np.asarray(x, dtype=np.float64)
        kmax = int(kmax)
        return self._higuchi_fd(x, kmax)

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


class KatzFractalDimension:
    """
    The Katz fractal dimension is yet another way to calculate the fractal dimension of a two-dimensional curve.
    """

    def katz_fd(self, x):
        x = np.array(x)
        dists = np.abs(np.ediff1d(x))
        ll = dists.sum()
        ln = np.log10(np.divide(ll, dists.mean()))
        aux_d = x - x[0]
        d = np.max(np.abs(aux_d[1:]))
        return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


def autoEDA(data, figsize=(16, 16)):

    sampling_rate = 24
    number_data_points = 10000

    resolution = [sampling_rate * 3, sampling_rate * 3]

    time_domain = np.arange(1 / sampling_rate, (number_data_points + 1) / sampling_rate, 1 / sampling_rate)
    _, ax = plt.subplots(4, len(data), figsize=figsize)
    for i in range(len(data)):
        ax[0, i].plot(time_domain, data[i], color='seagreen')
        ax[0, i].set_title('s1_{}_original'.format(i + 1), fontsize=24)
        ax[1, i].plot(time_domain, average_smoothing(data[i]), color='red')
        ax[1, i].set_title('s1_{}_avg_smooth'.format(i + 1), fontsize=24)
        ax[2, i].plot(time_domain, WaveletDenoising(data[i]).out, color='blue')
        ax[2, i].set_title('s1_{}_wavelet_denoise'.format(i + 1), fontsize=24)
        picked_data = WaveletDenoising(data[i]).out[:resolution[0] * resolution[1]].reshape(resolution)
        ax[3, i].imshow(picked_data)
        ax[3, i].set_title('s1_{}_feature_map'.format(i + 1), fontsize=24)

    plt.show()


if __name__ == '__main__':
    data = np.random.random(1000) - 0.5
    data_avg = average_smoothing(data)
    print(data_avg.shape)