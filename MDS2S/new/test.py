import numpy as np
import os
from config import Config as cfg
import matplotlib.pyplot as plt

file_names = os.listdir(cfg.IMFs_ROOT_DIR)
fn = file_names[0]
imfs = np.load(os.path.join(cfg.IMFs_ROOT_DIR, fn), allow_pickle=True)
t = cfg.T
plt.figure(figsize=(9, 6))
        # for i in range(3):
        #     plt.subplot(3, 1, i + 1)
        #     plt.plot(t[700:3500], imfs[i])
        # plt.show()
for i in cfg.CHANNELS:
    plt.subplot(3, 1, i + 1)
    plt.plot(t[700:3500], imfs.item().get('s{}'.format(i))[0])
plt.show()
# mc = 0
#
# for fn in file_names:
#     if fn == '.DS_Store':
#         continue
#     IMFs = np.load(os.path.join(cfg.IMFs_ROOT_DIR, fn), allow_pickle=True)
#     for i in range(3):
#         s = IMFs.item().get('s{}'.format(i))
#         mc = max(mc, np.max(s))
#
# print(mc)