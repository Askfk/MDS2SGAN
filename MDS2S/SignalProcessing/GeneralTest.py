import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

import scipy.io as scio
import os
import numpy as np


ROOT = ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'
file_names = os.listdir(ROOT_DIR)
file_name = '88-24-10000-400-6500-20-n.mat'
data_path = os.path.join(ROOT_DIR, file_name)
data = scio.loadmat(data_path)
t = np.arange(0, 10000 / 24000, 1 / 24000)
f = np.arange(0, 10000) * 24000 / 10000

s0 = data['s0'][:, 0]
s1 = data['s1'][:, 0]
s2 = data['s2'][:, 0]
s3 = data['s3'][:, 0]
s4 = data['s4'][:, 0]
s5 = data['s5'][:, 0]

data0 = [s0, s1, s2, s3, s4, s5]
plt.figure(figsize=(16, 6))

# 生成
# for i in range(6):
#     d = data[i]
#     ax = plt.subplot(2, 3, i+1)
#     l, = ax.plot(t, d, '.-')
#     ax.legend([l], ['通道S{}'.format(i)], loc='upper right')
#     ax.set_xlabel(u'时间')
#     ax.set_ylabel(u'振幅（mV）')
#
# plt.show()


