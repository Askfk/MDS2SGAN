import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import numpy as np
import scipy.io as scio
from TheRealEDA import FrequencyAnalysis, get_damage

"""
小波变换样例！后续就用这个
"""
sampling_rate = 24e3  # 采样频率
t = np.arange(0, 1e4/sampling_rate, 1.0/sampling_rate)

w = "cgau8"
totalscal = 3500 - 700
fc = pywt.central_frequency(w)  # 中心频率
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal, 1, -1)

damage = get_damage()
fa = FrequencyAnalysis(damage)
signals = fa.signals

figsize = (11, 8)

shape = (3, 2)

s17 = signals[17]
s25 = signals[25]

s17_temp = []
s25_temp = []
rgt = [700, 3500]
rgf = [2626, -29]

sLice = 144


plt.figure(figsize=figsize)
[cwtmatr, frequencies] = pywt.cwt(s17[0][rgt[0]: rgt[1]], scales, w, 1.0 / sampling_rate)  # 连续小波变换
for i in range((rgt[1] - rgt[0]) // sLice):
    plt.subplot(4, 5, i + 1)
    plt.contourf(t[sLice * i: sLice * (i+1)], frequencies[rgf[0]:rgf[1]], abs(cwtmatr)[rgf[0]:rgf[1], sLice * i: sLice * (i+1)])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Slice {}'.format(i+1))
plt.show()

# plt.figure(figsize=figsize)
# for i in range(1):
#     [cwtmatr, frequencies] = pywt.cwt(s17[i][rgt[0]: rgt[1]], scales, w, 1.0 / sampling_rate)  # 连续小波变换
#
#     plt.subplot(shape[0], shape[1], i + 1)
#     plt.contourf(t[rgt[0]: rgt[1]], frequencies[rgf[0]:rgf[1]], abs(cwtmatr)[rgf[0]:rgf[1]])
#     print(cwtmatr.shape)
#     plt.colorbar()
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.title('S{} Wavelet Magnitude'.format(i))
#     # plt.ylim([200, 700])
#     s17_temp.append([cwtmatr, frequencies])
#     print("17")
# plt.show()

# plt.figure(figsize=figsize)
# for i in range(6):
#     [cwtmatr, frequencies] = pywt.cwt(s25[i][rgt[0]: rgt[1]], scales, w, 1.0 / sampling_rate)  # 连续小波变换
#     plt.subplot(shape[0], shape[1], i + 1)
#     plt.contourf(t[rgt[0]: rgt[1]], frequencies[rgf[0]:rgf[1]], abs(cwtmatr)[rgf[0]:rgf[1]])
#     plt.colorbar()
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.title('S{} Wavelet Magnitude'.format(i))
#     # plt.ylim([200, 700])
#     print("25")
#     s25_temp.append([cwtmatr, frequencies])
# plt.show()
#
# plt.figure(figsize=figsize)
# for i in range(6):
#     cwt17, f17 = s17_temp[i]
#     cwt25, f25 = s25_temp[i]
#     f = (f17 + f25) / 2
#     cwt = np.abs(cwt17) - np.abs(cwt25)
#     plt.subplot(shape[0], shape[1], i + 1)
#     plt.contourf(t[rgt[0]: rgt[1]], f[rgf[0]:rgf[1]], abs(cwt)[rgf[0]:rgf[1]])
#     plt.colorbar()
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.title('S{} Wavelet Magnitude'.format(i))
#     # plt.ylim([200, 700])
#     print("17 - 25")
# plt.show()