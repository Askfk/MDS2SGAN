import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

LABEL_ROOT_DIR = "/Users/zcc/Documents/wield/labels"
RD = "/Users/zcc/Documents/wield/Loc&Depth"
IMFs_DIR = "/Users/zcc/Documents/wield/PickedIMFs"
IMFS = "/Users/zcc/Documents/wield/IMFs"
LD = "/Users/zcc/Documents/wield/Label_Corrected"

# file_names = os.listdir(IMFs_DIR)
#
# for fn in file_names:
#     if fn == '.DS_Store':
#         continue
#     data = np.load(os.path.join(IMFs_DIR, fn), allow_pickle=True)
#     np.save(os.path.join(IMFS, fn.split('-')[0]+'.npy'), data)
#
file_names = os.listdir(RD)

dealed = []

for fn in file_names:
    new = {}
    if fn == '.DS_Store':
        continue
    loc_dep = np.load(os.path.join(RD, fn), allow_pickle=True)
    loc = loc_dep.item().get('loc')
    depth = float(loc_dep.item().get('depth'))

    loc = np.pad(loc, ((0, 0), (17, 17)), 'constant').reshape([144, 80])[:, 4:-4]
    new['loc'] = loc
    new['depth'] = depth
    path = os.path.join(LD, fn)
    np.save(path, new)


