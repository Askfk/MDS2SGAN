import numpy as np
import matplotlib.pyplot as plt
import pywt
mode = pywt.Modes.smooth

from new.EDA import WaveletDenoising
from TheRealEDA import FrequencyAnalysis, get_damage

damage = get_damage()
fa = FrequencyAnalysis(damage)
signals = fa.signals
sampling_rate = 24e3  # 采样频率
t = np.arange(0, 1e4/sampling_rate, 1.0/sampling_rate)

rgt = (700, 3500)

s17 = signals[17]

plt.figure(figsize=(9, 3))
plt.plot(t, s17[0])
plt.show()

plt.figure(figsize=(9, 3))
d = WaveletDenoising(s17[0]).out
plt.plot(t, d)
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t[rgt[0]: rgt[1]], d[rgt[0]: rgt[1]])
plt.show()



