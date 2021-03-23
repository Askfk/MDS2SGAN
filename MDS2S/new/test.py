import numpy as np
import os
from config import Config as cfg

file_names = os.listdir(cfg.IMFs_ROOT_DIR)
mc = 0

for fn in file_names:
    if fn == '.DS_Store':
        continue
    IMFs = np.load(os.path.join(cfg.IMFs_ROOT_DIR, fn), allow_pickle=True)
    for i in range(3):
        s = IMFs.item().get('s{}'.format(i))
        mc = max(mc, np.max(s))

print(mc)