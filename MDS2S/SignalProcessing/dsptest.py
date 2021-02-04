import numpy as np
import matplotlib.pyplot as plt
import pywt

sampling_rate = 1024#采样频率
t = np.arange(0,1.0,1.0/sampling_rate)
f1 = 100#频率
f2 = 200
f3 = 300
data = np.piecewise(t,[t<1,t<0.8,t<0.3],
                    [lambda t : np.sin(2*np.pi *f1*t),
                     lambda t : np.sin(2 * np.pi * f2 * t),
                     lambda t : np.sin(2 * np.pi * f3 * t)])
wavename = "cgau8"
totalscal = 10000
fc = pywt.central_frequency(wavename)#中心频率
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal,1,-1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0/sampling_rate) # 连续小波变换
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.show()
