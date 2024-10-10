# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 下午2:39
# @Author  : Guo Huimin
import random

import numpy as np
import torch


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalized data
    """
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def map_color(image, class_num=3, gray_flag=False):
    """map the prediction/label to the RGB space.

    Parameters
    ----------
    image : _type_
        _description_
    class_num : _type_
        _description_
    gray_flag : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    cmap = np.zeros(shape=(class_num, 3)).astype(np.uint8)
    cmap[0, :] = np.array([0, 0, 0])
    for class_ix in range(1, class_num):
        cmap[class_ix, :] = np.array([0, 255, 0])  # green for foreground

    if gray_flag:
        return (image * 255).astype(np.uint8)
    else:
        final_image = np.zeros(shape=(3, image.shape[0], image.shape[1])) * 255
        for label in range(0, class_num):
            mask = image == label
            final_image[0][mask] = cmap[label][0]
            final_image[1][mask] = cmap[label][1]
            final_image[2][mask] = cmap[label][2]
        return normalize(final_image)


def show4tb_2d(inputs, labels, outputs, phase="train"):
    # inputs, labels, outputs -> bcxyz
    b = int(random.sample(range(inputs.shape[0]), 1)[0])
    inputs = inputs[b, 0, ...].cpu().numpy()  # xyz

    labels = labels[b]  # cxy c=2 if val else 11
    outputs = outputs[b]  # cxy c=2

    if phase != "train":
        labels = torch.argmax(labels, dim=0).cpu().numpy()  # xy
    else:
        labels = labels[0].cpu().numpy()
    outputs = torch.argmax(outputs, dim=0).cpu().numpy()  # xy

    # yx
    inputs = np.transpose(inputs, (1, 0))
    labels = np.transpose(labels, (1, 0))
    outputs = np.transpose(outputs, (1, 0))
    return inputs, labels, outputs
