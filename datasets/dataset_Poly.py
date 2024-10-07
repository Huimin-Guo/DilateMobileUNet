import os
import pickle

import numpy as np
from monai.data.image_reader import PILReader
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld,
                              RandFlipd, RandRotate90d, ScaleIntensityd)

# pre-defined functions
listdir = os.listdir
join = os.path.join
path_exits = os.path.exists
mkdir = os.mkdir
mkdir_p = os.makedirs


def subfiles(rootPath: str) -> list:
    """find all files under the root path, and return rooPath/fileName automatically.

    Parameters
    ----------
    rootPath : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    files = os.listdir(rootPath)
    filesList = [os.path.join(rootPath, ix) for ix in files]
    return filesList


# cross-validation generation
# split the dataset into train/test set
class fold_generator:
    def __init__(self, seed, n_splits, len_data):
        """
        :param seed: Random seed for splits.
        :param n_splits: number of splits, e.g. 5 splits for 5-fold cross-validation
        :param len_data: number of elements in the dataset.
        """
        self.tr_ix = []
        self.val_ix = []
        self.te_ix = []
        self.slicer = None
        self.missing = 0
        self.fold = 0
        self.len_data = len_data
        self.n_splits = n_splits
        self.myseed = seed
        self.boost_val = 0

    def init_indices(self):

        t = list(np.arange(self.l))
        # round up to next splittable data amount.
        split_length = int(np.ceil(len(t) / float(self.n_splits)))
        self.slicer = split_length
        self.mod = len(t) % self.n_splits
        if self.mod > 0:
            # missing is the number of folds, in which the new splits are reduced to account for missing data.
            self.missing = self.n_splits - self.mod

        self.te_ix = t[:self.slicer]
        self.tr_ix = t[self.slicer:]
        self.val_ix = self.tr_ix[:self.slicer]
        self.tr_ix = self.tr_ix[self.slicer:]

    def new_fold(self):

        slicer = self.slicer
        if self.fold < self.missing:
            slicer = self.slicer - 1

        temp = self.te_ix

        # catch exception mod == 1: test set collects 1+ data since walk through both roudned up splits.
        # account for by reducing last fold split by 1.
        if self.fold == self.n_splits - 2 and self.mod == 1:
            temp += self.val_ix[-1:]
            self.val_ix = self.val_ix[:-1]

        self.te_ix = self.val_ix
        self.val_ix = self.tr_ix[:slicer]
        self.tr_ix = self.tr_ix[slicer:] + temp

    def get_fold_names(self):
        names_list = []
        rgen = np.random.RandomState(self.myseed)
        cv_names = np.arange(self.len_data)

        rgen.shuffle(cv_names)
        self.l = len(cv_names)
        self.init_indices()

        for split in range(self.n_splits):
            train_names, val_names, test_names = cv_names[
                self.tr_ix], cv_names[self.val_ix], cv_names[self.te_ix]
            names_list.append([train_names, val_names, test_names, self.fold])
            self.new_fold()
            self.fold += 1

        return names_list


def cv_5(input, output, saved_name):
    paths_image = subfiles(join(input, "image"))
    imageNames = listdir(join(input, "image"))
    if not path_exits(join(output, f"{saved_name}.pickle")):
        len_data = len(listdir(join(input, "image")))
        fg = fold_generator(seed=0, n_splits=5, len_data=len_data)
        fg_lists = fg.get_fold_names()

        cv_plan = {i: {} for i in range(5)}
        for fold, fg_list in enumerate(fg_lists):
            ix_train, ix_val, ix_test = fg_list[0].tolist(
            ), fg_list[1].tolist(), fg_list[2].tolist()
            cv_plan[fold]["train"], cv_plan[fold]["val"], cv_plan[fold]["test"] = [
            ], [], []
            for ix in ix_train:
                name = imageNames[ix]
                cv_plan[fold]["train"].append(
                    {"image": str(name), "label": str(name)})
            for ix in ix_val:
                name = imageNames[ix]
                cv_plan[fold]["val"].append(
                    {"image": str(name), "label": str(name)})
            for ix in ix_test:
                name = imageNames[ix]
                cv_plan[fold]["test"].append(
                    {"image": str(name), "label": str(name)})
        with open(join(output, f"{saved_name}.pickle"), 'wb') as f:
            pickle.dump(cv_plan, f)
    else:
        print("you have created a cross-validation plan pickle.")


# ----------------------------------------------------------------------------------------------------------------------
def image_label_dict(root_dir, image_names):
    """
    return a dict including the image/label pairs.
    Parameters
    ----------
    root_dir
    image_names

    Returns
    -------

    """
    image_list = [join(root_dir, 'image', image_name)
                  for image_name in image_names]
    label_list = [join(root_dir, 'label', image_name)
                  for image_name in image_names]
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(image_list, label_list)
    ]
    return data_dicts


def complete_cv5_names(sets, image_names):
    image_2d_names = listdir(join(sets.root_dir, "image"))

    new_image_names = []
    for image_2d_name in image_2d_names:
        pure_image_name = f"{'_'.join(image_2d_name.split('_')[:-1])}.nii.gz"
        if pure_image_name in image_names:
            new_image_names.append(image_2d_name)
    return new_image_names


def prepare_my_data_poly(sets):
    # create/load cross validation file
    path_cv_plan = join(sets.root_dir, f"{sets.cv5_name}.pickle")
    try:
        assert path_exits(path_cv_plan), "there is no cross-validation plan."
    except:
        print("Start generating the cross validation pickle file ...")
        cv_5(sets.root_dir, sets.root_dir, sets.cv5_name)
    with open(path_cv_plan, 'rb') as f:
        cv_plan = pickle.load(f)
    print(f"cross validation is loaded from {path_cv_plan}")

    train_image_names = [data["image"] for data in cv_plan[sets.fold]["train"]]
    val_image_names = [data["image"] for data in cv_plan[sets.fold]["val"]]
    test_image_names = [data["image"] for data in cv_plan[sets.fold]["test"]]

    train_dict = image_label_dict(sets.root_dir, train_image_names)
    val_dict = image_label_dict(sets.root_dir, val_image_names)
    test_dict = image_label_dict(sets.root_dir, test_image_names)

    if sets.phase != 'train':
        return train_dict, val_dict, test_dict
    else:
        train_transforms = Compose([
            LoadImaged(keys=["image"]),
            LoadImaged(keys=["label"], reader=PILReader(
                converter=lambda label: label.convert("L"))),
            EnsureChannelFirstd(keys=["image", "label"], ),
            ScaleIntensityd(keys=["image", "label"]),
            RandCropByPosNegLabeld(keys=["image", "label"],
                                   label_key="label",
                                   spatial_size=tuple(sets.patch_size),
                                   pos=1,
                                   neg=1,
                                   num_samples=4,
                                   ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.25,
                max_k=3,
            ),
        ])
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                LoadImaged(keys=["label"], reader=PILReader(
                    converter=lambda label: label.convert("L"))),
                EnsureChannelFirstd(keys=["image", "label"], ),
                ScaleIntensityd(keys=["image", "label"]),
            ])
        return train_dict, val_dict, test_dict, train_transforms, val_transforms
