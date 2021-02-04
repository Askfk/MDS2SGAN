import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import Config as cfg


TEMP_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/temp'

file_names = os.listdir(TEMP_DIR)

fn1 = file_names[0].split('-')[-1][:-4]
params = list(map(int, fn1.split()))
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

plt.figure(figsize=(45, 35))

plt.subplot(4, 4, 1)
sns.heatmap(label, annot=False)

i = 2
for fn in file_names:
    label = np.loadtxt(os.path.join(TEMP_DIR, fn))
    plt.subplot(4, 4, i)
    sns.heatmap(label, annot=False)
    i += 1

plt.show()