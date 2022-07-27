import cv2
import numpy as np
import os.path as osp

CLASSES = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
           'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
PALETTE = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
           [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
           [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
# plt.show()


import os
from os import listdir
from os.path import isfile, join
from PIL import Image


def findindex(key_color, PALETTE):
    for i, color in enumerate(PALETTE):
        if key_color[0] == PALETTE[i][0] and key_color[1] == PALETTE[i][1] and key_color[2] == PALETTE[i][2]:
            return i
    return -1


def get_number_for_classes(mask_dir='AI6126_dataset_public/train/train_mask'):
    mask_dir = 'AI6126_dataset_public/train/train_mask'
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if
                  os.path.isfile(os.path.join(mask_dir, f))]

    # List[int] 即 [0,0]
    counts_per_color = [0] * 19

    for index, file in enumerate(mask_files):
        print(index, file)
        img = Image.open(file).convert('RGB')
        na = np.array(img)
        colours, counts = np.unique(na.reshape(-1, 3), axis=0, return_counts=1)
        print(colours)
        print(counts)
        for i, color in enumerate(colours):
            index = findindex(color, PALETTE)
            print(color, index)
            counts_per_color[index] += counts[i]
        print(counts_per_color)

    return counts_per_color


# 通过计算正负样本的数量、比例, 给权重
def make_weights_for_balanced_classes(label_list, nclasses):
    count = label_list
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print("number of pixels, N: ", N)

    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    print("weight_per_class: ", weight_per_class)
    return weight_per_class


def getWeight(counts_per_color, counts_class=19):
    weight_per_class = make_weights_for_balanced_classes(counts_per_color, counts_class)
    weights = np.array(weight_per_class)
    weights = np.log2(weights)
    weights = np.round(weights, 4)
    return weights.tolist()

# log2:  [1.805, 1.976, 5.6004, 8.5621, 8.8014, 8.8059, 7.8778, 7.9143, 7.7323, 8.0199, 8.3939, 7.9185, 7.2026, 1.6725, 6.695, 8.7228, 12.7002, 4.6004, 4.8802]
# log10: [0.5434, 0.5948, 1.6859, 2.5774, 2.6495, 2.6508, 2.3715, 2.3824, 2.3276, 2.4142, 2.5268, 2.3837, 2.1682, 0.5035, 2.0154, 2.6258, 3.8231, 1.3849, 1.4691]
# log1p: [1.5028, 1.5962, 3.9023, 5.9374, 6.1029, 6.106, 5.4647, 5.4899, 5.3643, 5.5628, 5.8212, 5.4928, 4.9992, 1.4322, 4.6502, 6.0485, 8.8033, 3.2292, 3.4161]
# emath.log: [1.2511, 1.3697, 3.8819, 5.9348, 6.1007, 6.1038, 5.4605, 5.4858, 5.3596, 5.559, 5.8182, 5.4887, 4.9925, 1.1593, 4.6406, 6.0462, 8.8031, 3.1888, 3.3827]


if __name__ == '__main__':
    counts_per_color = get_number_for_classes()
    # counts_per_color = [375099645, 333172471, 27015905, 3467864, 2937773, 2928618, 5572580, 5433404, 6164110, 5049695,
    #                     3896683, 5417555, 8898431, 411175974, 12651045, 3102372, 196952, 54031542, 44507381]

    weights = getWeight(counts_per_color, 19)
    print('log', weights)

