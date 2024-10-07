# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 下午2:39
# @Author  : Guo Huimin
# @Email   : guohuimin2619@foxmail.com
# @FileName: evaluate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from einops import rearrange
import numpy as np
import random


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:,
                           channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=False)


class dice_train():
    def __init__(self, sets, include_background=False):
        # axial, sag, and cor planes
        self.inner_mode_list = [
            'b c x y z -> b c x y z',
            'b c x y z -> b c x z y',
            'b c x y z -> b c z y x',
        ]
        self.sets = sets
        self.dice_metric = multiclass_dice_coeff if self.sets.n_seg_classes > 2 else dice_coeff
        self.include_background = include_background

    def dice(self, views, label):
        label = F.one_hot(label[:, 0, ...].to(dtype=torch.long), num_classes=self.sets.n_seg_classes).permute(0, 4, 1,
                                                                                                              2,
                                                                                                              3).float()
        views = [F.one_hot(view.argmax(dim=1), self.sets.n_seg_classes).permute(0, 4, 1, 2, 3).float() for view in
                 views]
        dice = 0
        for ix, view in enumerate(views):
            label = rearrange(label, self.inner_mode_list[ix])
            # background is not included
            if self.include_background:
                dice_view = self.dice_metric(view, label)
            else:
                dice_view = self.dice_metric(
                    view[:, 1:, ...], label[:, 1:, ...])
            dice += dice_view
        dice /= len(views)
        return dice


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


def show4tb(inputs, labels, outputs, phase="train"):
    # inputs, labels, outputs -> bcxyz
    b = int(random.sample(range(inputs.shape[0]), 1)[0])
    inputs = inputs[b, 0, ...].cpu().numpy()  # xyz

    labels = labels[b]  # cxyz c=2 if val else 11
    outputs = outputs[b]  # cxyz c=2

    if phase != "train":
        labels = torch.argmax(labels, dim=0).cpu().numpy()  # xyz
    else:
        labels = labels[0].cpu().numpy()
    outputs = torch.argmax(outputs, dim=0).cpu().numpy()  # xyz

    z_index = list(set(np.where(labels != 0)[-1].tolist()))
    if len(z_index) != 0:
        # randomly sample one z_index from the list
        z_index = int(random.sample(z_index, 1)[0])
        # z_index = z_index[int(len(z_index) / 2)]
    else:
        z_index = int(random.sample(range(labels.shape[-1]), 1)[0])

    # yx
    inputs = np.transpose(inputs[..., z_index], (1, 0))
    labels = np.transpose(labels[..., z_index], (1, 0))
    outputs = np.transpose(outputs[..., z_index], (1, 0))
    return inputs, labels, outputs


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
