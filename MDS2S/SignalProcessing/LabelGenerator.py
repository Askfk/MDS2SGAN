import os
import numpy as np
from ..config import Config as config
import matplotlib.pyplot as plt
import seaborn as sns


def generate_location_labels(height=config.LOCAL_HEIGHT, width=config.LOCAL_WIDTH, name='-24-12000-400-6500-20-',
                             i=0):
    ROOT = "/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/loaction_labels"
    print("Enter type:")
    type = input()

    params = list(map(int, type.split()))

    [x1, y1] = [params[0], params[1]]

    [x3, y3] = [params[2], params[3]]
    [x2, y2] = [x3, y1]
    [x4, y4] = [x1, y3]

    num = params[-1]

    end = i + num
    while i < end:
        tname = str(i) + name + 'hv-' + type
        label = np.zeros([height, width])

        tx1 = max(0, min(config.LOCAL_WIDTH, x1 + np.random.randint(-3, 3, 1)[0]))
        ty1 = max(0, min(config.LOCAL_HEIGHT, y1 + np.random.randint(-3, 3, 1)[0]))

        tx2 = max(0, min(config.LOCAL_WIDTH, x2 + np.random.randint(-3, 3, 1)[0]))
        ty2 = max(0, min(config.LOCAL_HEIGHT, y2 + np.random.randint(-3, 3, 1)[0]))

        tx3 = max(0, min(config.LOCAL_WIDTH, x3 + np.random.randint(-3, 3, 1)[0]))
        ty3 = max(0, min(config.LOCAL_HEIGHT, y3 + np.random.randint(-3, 3, 1)[0]))

        tx4 = max(0, min(config.LOCAL_WIDTH, x4 + np.random.randint(-3, 3, 1)[0]))
        ty4 = max(0, min(config.LOCAL_HEIGHT, y4 + np.random.randint(-3, 3, 1)[0]))

        if tx1 >= tx3 or ty2 >= ty4:
            print("ERROR JUMP")
            continue

        if (tx2 - tx1) == 0:
            continue
        upk = (ty2 - ty1) / (tx2 - tx1)
        upb = y1 - upk * x1

        if (tx3 - tx2) == 0:
            continue
        rightk = (ty3 - ty2) / (tx3 - tx2)
        rightb = y2 - rightk * x2

        if (tx4 - tx3) == 0:
            continue
        downk = (ty4 - ty3) / (tx4 - tx3)
        downb = y3 - downk * x3

        if (tx1 - tx4) == 0:
            continue
        leftk = (ty1 - ty4) / (tx1 - tx4)
        leftb = y4 - leftk * x4

        sx, ex = tx1, tx3
        sy, ey = ty2, ty4
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

        if not np.any(label == 1):
            print("{} occurs ERROR, skip it".format(tname))
            continue

        save_path = os.path.join(ROOT, tname + '.txt')
        f = open(save_path, 'w')
        f.close()
        np.savetxt(save_path, label)
        print('Done saving {}'.format(tname))

        sns.heatmap(label, annot=False)
        plt.title(tname)
        plt.show()

        i += 1

    print('------------------------Split Line----------------------------')
    return i


def generate_local_label(height=config.LOCAL_HEIGHT, width=config.LOCAL_WIDTH, name='-24-12000-400-6500-20-', j=0):
    ROOT = "/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/loaction_labels"
    print("Enter type:")
    type = input()

    params = list(map(int, type.split()))

    [x1, y1] = [params[0], params[1]]

    [x2, y2] = [params[2], params[3]]

    [x3, y3] = [params[6], params[7]]

    [x4, y4] = [params[4], params[5]]

    num = params[-1]

    end = j + num
    while j < end:
        tname = str(j) + name + type
        label = np.zeros([height, width])

        tx1 = max(0, min(config.LOCAL_WIDTH, x1 + np.random.randint(-3, 3, 1)[0]))
        ty1 = max(0, min(config.LOCAL_HEIGHT, y1 + np.random.randint(-3, 3, 1)[0]))

        tx2 = max(0, min(config.LOCAL_WIDTH, x2 + np.random.randint(-3, 3, 1)[0]))
        ty2 = max(0, min(config.LOCAL_HEIGHT, y2 + np.random.randint(-3, 3, 1)[0]))

        tx3 = max(0, min(config.LOCAL_WIDTH, x3 + np.random.randint(-3, 3, 1)[0]))
        ty3 = max(0, min(config.LOCAL_HEIGHT, y3 + np.random.randint(-3, 3, 1)[0]))

        tx4 = max(0, min(config.LOCAL_WIDTH, x4 + np.random.randint(-3, 3, 1)[0]))
        ty4 = max(0, min(config.LOCAL_HEIGHT, y4 + np.random.randint(-3, 3, 1)[0]))

        if tx1 >= tx3 or ty2 >= ty4:
            print("ERROR JUMP")
            continue

        if (tx2 - tx1) == 0:
            continue
        upk = (ty2 - ty1) / (tx2 - tx1)
        upb = y1 - upk * x1

        if (tx3 - tx2) == 0:
            continue
        rightk = (ty3 - ty2) / (tx3 - tx2)
        rightb = y2 - rightk * x2

        if (tx4 - tx3) == 0:
            continue
        downk = (ty4 - ty3) / (tx4 - tx3)
        downb = y3 - downk * x3

        if (tx1 - tx4) == 0:
            continue
        leftk = (ty1 - ty4) / (tx1 - tx4)
        leftb = y4 - leftk * x4

        sx, ex = tx1, tx3
        sy, ey = ty2, ty4
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

        if not np.any(label == 1):
            print("{} occurs ERROR, skip it".format(tname))
            continue

        save_path = os.path.join(ROOT, tname + '.txt')
        f = open(save_path, 'w')
        f.close()
        np.savetxt(save_path, label)
        print('Done saving {}'.format(tname))

        sns.heatmap(label, annot=False)
        plt.title(tname)
        plt.show()
        j += 1

    print("-----------------------Finished this loop------------------------")

    return j


i = 401
while i != -1:
    print("Is horizontal or vertical? (y/n):")
    tp = input()

    if tp == 'n':
        i = generate_local_label(j=i)
    else:
        i = generate_location_labels(i=i)
# label = np.loadtxt('/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/test/2_test_t.txt')
# sns.heatmap(label, annot=False)
# plt.show()