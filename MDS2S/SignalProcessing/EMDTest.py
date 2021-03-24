import numpy as np
import matplotlib.pyplot as plt

# from ..config import Config as cfg
#
#
# TEMP_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/temp'
#
# file_names = os.listdir(TEMP_DIR)
#
# fn1 = file_names[0].split('-')[-1][:-4]
# params = list(map(int, fn1.split()))
# [x1, y1] = [params[0], params[1]]
# [x2, y2] = [params[2], params[3]]
# [x3, y3] = [params[6], params[7]]
# [x4, y4] = [params[4], params[5]]
#
# upk = (y2 - y1) / (x2 - x1)
# upb = y1 - upk * x1
# rightk = (y3 - y2) / (x3 - x2)
# rightb = y2 - rightk * x2
# downk = (y4 - y3) / (x4 - x3)
# downb = y3 - downk * x3
# leftk = (y1 - y4) / (x1 - x4)
# leftb = y4 - leftk * x4
# sx, ex = x1, x3
# sy, ey = y2, y4
# label = np.zeros([cfg.LOCAL_HEIGHT, cfg.LOCAL_WIDTH])
# for tx in range(sx, ex):
#     for ty in range(sy, ey):
#         if ty >= upk * tx + upb:
#             if ty >= rightk * tx + rightb:
#                 if ty <= downk * tx + downb:
#                     if ty <= leftk * tx + leftb:
#                         label[ty, tx] = 1
#                     else:
#                         continue
#                 else:
#                     continue
#             else:
#                 continue
#         else:
#             continue
#
# plt.figure(figsize=(45, 35))
#
# plt.subplot(4, 4, 1)
# sns.heatmap(label, annot=False)
#
# i = 2
# for fn in file_names:
#     label = np.loadtxt(os.path.join(TEMP_DIR, fn))
#     plt.subplot(4, 4, i)
#     sns.heatmap(label, annot=False)
#     i += 1
#
# plt.show()

import scipy.io as scio
import os
from PyEMD import EMD, Visualisation
from new.EDA import WaveletDenoising

ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'

file_name = '165-24-10000-400-6500-20-n.mat'
title = 'EMD'
fn = os.path.join(ROOT_DIR, file_name)

data = scio.loadmat(fn)
t = np.arange(0, 10000/24000, 1/24000)
f = np.arange(0, 10000) * 24000 / 10000

s0 = data['s0'][:, 0]
s1 = data['s1'][:, 0]
s2 = data['s2'][:, 0]
s3 = data['s3'][:, 0]
s4 = data['s4'][:, 0]
s5 = data['s5'][:, 0]

d = WaveletDenoising(s1).out

emd = EMD()
emd.emd(d, T=t)
imfs, res = emd.get_imfs_and_residue()

fig = plt.figure(figsize=(12, 16))
ax_main = fig.add_subplot(len(imfs) + 1, 1, 1)
ax_main.set_title(title)
ax_main.plot(t, d)

rec_b = []
for i in imfs:
    rec_b.append(d-i)
    d -= i

for i, y in enumerate(rec_b):
    ax = fig.add_subplot(len(rec_b) + 1, 2, 3 + i * 2)
    ax.plot(t, y, 'r')

    ax.set_ylabel("A%d" % (i + 1))

for i, y in enumerate(imfs):
    ax = fig.add_subplot(len(imfs) + 1, 2, 4 + i * 2)
    ax.plot(t, y, 'g')
    ax.set_ylabel("D%d" % (i + 1))


# In general:
#components = EEMD()(S)
#imfs, res = components[:-1], components[-1]

vis = Visualisation()
vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
vis.plot_instant_freq(f, imfs=imfs)
vis.show()