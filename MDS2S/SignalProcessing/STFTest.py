import scipy.signal as scignal
import os
import numpy as np
import matplotlib.pyplot as plt

from TheRealEDA import FrequencyAnalysis, get_damage

damage = get_damage()
fa = FrequencyAnalysis(damage)

signals = fa.signals
figsize = (12, 8)

show_range = (0, 800)

nperseg = 128
shape = (3, 2)

s17 = signals[17]
s25 = signals[25]


s17_temp = []
plt.figure(figsize=figsize)
for j in range(len(s17)):
    params = scignal.stft(s17[j], fs=24e3, nperseg=nperseg)
    f, t, zxx = params
    plt.subplot(shape[0], shape[1], j + 1)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('S{} STFT Magnitude'.format(j))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(show_range[0], show_range[1])
    s17_temp.append(params)
plt.show()

s25_temp = []
plt.figure(figsize=figsize)
for j in range(len(s25)):
    params = scignal.stft(s25[j], fs=24e3, nperseg=nperseg)
    f, t, zxx = params
    plt.subplot(shape[0], shape[1], j + 1)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('S{} STFT Magnitude'.format(j))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(show_range[0], show_range[1])
    s25_temp.append(params)
plt.show()

plt.figure(figsize=figsize)
for i in range(6):
    f17, t17, zxx17 = s17_temp[i]
    f25, t25, zxx25 = s25_temp[i]
    t = (t17 + t25) / 2
    f = (f17 + f25) / 2
    zxx = np.abs(zxx17) - np.abs(zxx25)

    plt.subplot(shape[0], shape[1], i + 1)
    plt.pcolormesh(t, f, np.abs(zxx))
    plt.colorbar()
    plt.title('S{} STFT Magnitude'.format(j))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(show_range[0], show_range[1])
plt.show()
