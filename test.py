import os
import pdb
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from monai.data import (CacheDataset, DataLoader, Dataset, decollate_batch,
                        pad_list_data_collate)
from monai.data.image_reader import PILReader
from monai.data.image_writer import PILWriter
from monai.handlers.utils import from_engine
from monai.inferers import SliceInferer, sliding_window_inference
from monai.metrics import ConfusionMatrixMetric, DiceMetric, MeanIoU
from monai.transforms import (Activations, Activationsd, AsDiscrete,
                              AsDiscreted, Compose, CropForegroundd,
                              EnsureChannelFirstd, EnsureTyped, Invert,
                              Invertd, LoadImaged, NormalizeIntensityd,
                              Orientationd, SaveImaged, ScaleIntensityd,
                              ScaleIntensityRanged, Spacingd)
from monai.transforms.utils import allow_missing_keys_mode
from tqdm import tqdm

from datasets.dataset_Poly import prepare_my_data_poly
from models.DilateMobileUnet import DilateMobileVitUnet
from setting import parse_opts


def main_2d():
    model.eval()
    model.phase = "test"
    cuda_device, cpu_device = torch.device(f"cuda:{sets.gpu_id}"), torch.device("cpu")
    crf_transforms = AsDiscrete(to_onehot=sets.n_seg_classes)
    with torch.no_grad():
        for val_data in tqdm(data_loader):
            val_inputs = val_data["image"].to(cuda_device)
            roi_size = tuple(sets.patch_size)  # list to tuple
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,
                                                        overlap=sets.patch_overlap,
                                                        device=cuda_device
                                                        )
            # get the image name
            val_name = val_data["image"].meta['filename_or_obj'][0]

            # start calculating dice scores
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(
                val_data)  # compute metric for current iteration
            val_labels = [i.to(cuda_device) for i in val_labels]

            # save the prediction
            pred_show = torch.argmax(val_outputs[0], dim=0)
            pred_show = np.transpose(
                pred_show.as_tensor().cpu().numpy(), axes=(1, 0))
            cv2.imwrite(
                f"{prediction_folder}/{str(Path(val_name).name)}", pred_show*255)

            dice_val_outputs = dice_metric(y_pred=val_outputs, y=val_labels)
            iou_val_outputs = mean_iou_metric(y_pred=val_outputs, y=val_labels)
            confusion_val_outputs = confusion_metric(
                y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        dice_metric_org = dice_metric.aggregate().item()
        iou_metric_org = mean_iou_metric.aggregate().item()
        confusion_metric_org = confusion_metric.aggregate()
        # reset the status for next validation round
        dice_metric.reset()
        mean_iou_metric.reset()
        confusion_metric.reset()

    print("Dice Metric: ", dice_metric_org)
    print("IoU Metric: ", iou_metric_org)
    print("Recall, Precision, Accuracy: ", confusion_metric_org)


if __name__ == "__main__":
    sets = parse_opts()

    np.random.seed(sets.manual_seed)
    torch.manual_seed(sets.manual_seed)
    torch.cuda.manual_seed_all(sets.manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{sets.gpu_id}")
    checkpoint = torch.load(sets.resume_path)

    # model
    if sets.model == "dilatemobile":
        model = DilateMobileVitUnet(
            sets=sets, spatial_dims=2, in_channels=sets.n_input_channels, out_channels=sets.n_seg_classes).to(device)
    else:
        raise NotImplementedError

    model.load_state_dict(checkpoint['state_dict'])

    if not sets.generalization:
        prediction_type = str(Path(sets.resume_path).name).split("_")[0]
        prediction_folder = os.path.join(
            str(Path(sets.resume_path).parent.parent), f"prediction_{prediction_type}")
    else:
        prediction_folder = f"prediction_generalization/{sets.task_name}"
    if not os.path.exists(prediction_folder):
        os.makedirs(prediction_folder)
    print(f"prediction folder -> {prediction_folder}")

    test_transforms = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        LoadImaged(keys=["label"], reader=PILReader(
            converter=lambda label: label.convert("L"))),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),

        EnsureTyped(keys=["image", "label"]),
    ])

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(keys="pred", transform=test_transforms, orig_keys="image",
                meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
                ),
        # the AsDiscreted is for saving prediction
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),

        # following AsDiscreted is for calculating dice
        AsDiscreted(
            keys="pred", to_onehot=sets.n_seg_classes),
        AsDiscreted(
            keys="label", to_onehot=sets.n_seg_classes),
    ])

    cv_name = sets.cv5_name.split("_")[0]
    if cv_name == "poly":
        train_dict, val_dict, test_dict = prepare_my_data_poly(
            sets)
    else:
        raise NotImplementedError

    test_ds = CacheDataset(data=test_dict, transform=test_transforms, cache_rate=1)
    data_loader = DataLoader(test_ds, shuffle=False,
                             batch_size=1, num_workers=sets.num_workers, drop_last=False)

    include_background = False
    dice_metric = DiceMetric(
        include_background=include_background, reduction="mean")
    mean_iou_metric = MeanIoU(
        include_background=include_background, reduction="mean")
    confusion_metric = ConfusionMatrixMetric(include_background=include_background, reduction="mean", metric_name=[
                                             "sensitivity", "precision", "accuracy"])

    main_2d()
