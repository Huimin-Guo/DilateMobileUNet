import os
import pdb

import monai
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataset_Poly import prepare_my_data_poly
from models.DilateMobileUnet import DilateMobileVitUnet
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from setting import parse_opts
from utils import map_color, show4tb, show4tb_2d


def valid_2d(sets, val_inputs, val_labels, post_pred, post_label, device, cpu_device):
    roi_size = tuple(sets.patch_size)  # list to tuple
    sw_batch_size = 4
    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,
                                           sw_device=device, device=cpu_device, overlap=sets.patch_overlap, mode="gaussian")
    val_outputs = [post_pred(i)
                   for i in decollate_batch(val_outputs)]
    val_labels = [post_label(i)
                  for i in decollate_batch(val_labels)]

    val_outputs = [val_output.to(
        cpu_device) for val_output in val_outputs]
    val_labels = [val_label.to(cpu_device)
                  for val_label in val_labels]
    return val_outputs, val_labels


def train(sets, data_loader, model, current_epoch, writer, checkpoints_path, best_metric_epoch, best_metric):
    max_epochs = sets.n_epochs
    metric_values = []

    post_pred = AsDiscrete(argmax=True, to_onehot=sets.n_seg_classes)
    post_label = AsDiscrete(to_onehot=sets.n_seg_classes)

    train_loader, val_loader = data_loader[0], data_loader[1]

    for epoch in range(current_epoch, max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        model.phase = "train"
        epoch_loss = 0
        epoch_loss_val = 0
        epoch_dice = 0
        step = 0
        for train_ix, batch_data in enumerate(train_loader):
            # with torch.autograd.set_detect_anomaly(True):
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)

            # loss
            loss = loss_function(outputs, labels)

            epoch_loss += loss.item()

            # dice
            outputs_for_dice = F.one_hot(torch.argmax(torch.softmax(outputs, 1), 1),
                                         num_classes=sets.n_seg_classes).permute(0, 3, 1, 2)
            labels_for_dice = F.one_hot(labels[:, 0, ...].to(dtype=torch.long), num_classes=sets.n_seg_classes).permute(
                0, 3, 1, 2)
            dice_metric_train(y_pred=outputs_for_dice, y=labels_for_dice)
            dice = dice_metric_train.aggregate().item()
            epoch_dice += dice
            dice_metric_train.reset()

            # update parameters
            loss.backward()
            optimizer.step()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size} | epoch {epoch + 1}, "
                f"train_loss: {loss.item():.4f} train_dice: {dice:.4f}")
            if train_ix % 10 == 0:
                show_image, show_label, show_prediction = show4tb_2d(
                    inputs, labels, outputs)
                writer.add_image('Train/image', map_color(show_image.astype(float), gray_flag=True),
                                 step + epoch * len(train_ds), dataformats='HW')
                writer.add_image('Train/label', map_color(show_label, class_num=sets.n_seg_classes),
                                 step + epoch * len(train_ds),
                                 dataformats='CHW')
                writer.add_image('Train/prediction', map_color(show_prediction, class_num=sets.n_seg_classes),
                                 step + epoch * len(train_ds), dataformats='CHW')
        scheduler.step()
        epoch_loss /= step
        epoch_dice /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        print(f"epoch {epoch + 1} average dice: {epoch_dice:.4f}")
        writer.add_scalar("Metric/Loss", epoch_loss, epoch)
        writer.add_scalar("Metric/Dice", epoch_dice, epoch)
        writer.add_scalar("Metric/Lr", optimizer.param_groups[0]['lr'], epoch)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            os.path.join(checkpoints_path, "last_model.pth"))

        if sets.val:
            if (epoch + 1) % sets.val_interval == 0:
                model.eval()
                model.phase = "test"
                with torch.no_grad():
                    pbar = tqdm(val_loader)
                    for val_ix, val_data in enumerate(pbar):
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(cpu_device),
                        )

                        val_outputs, val_labels = valid_2d(
                                sets, val_inputs, val_labels, post_pred, post_label, device, cpu_device)
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        loss_val = loss_function(torch.unsqueeze(
                            val_outputs[0], 0), val_data["label"])
                        epoch_loss_val += loss_val.item()

                        if val_ix % 10 == 0:
                            show_image, show_label, show_prediction = show4tb_2d(val_inputs, val_labels, val_outputs,
                                                                                     phase="test")
                            writer.add_image('Val/image', map_color(show_image.astype(float), gray_flag=True),
                                             (val_ix + 1) + epoch * len(val_ds), dataformats='HW')
                            writer.add_image('Val/label', map_color(show_label, class_num=sets.n_seg_classes),
                                             (val_ix + 1) +
                                             epoch * len(val_ds),
                                             dataformats='CHW')
                            writer.add_image('Val/prediction', map_color(show_prediction, class_num=sets.n_seg_classes),
                                             (val_ix + 1) + epoch * len(val_ds), dataformats='CHW')

                    # validation loss
                    epoch_loss_val /= len(val_loader)
                    writer.add_scalar("Metric/Val_Loss", epoch_loss_val, epoch)
                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    writer.add_scalar("Metric/Val_Dice", metric, epoch)
                    # reset the status for next validation round
                    dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict()
                        },
                            os.path.join(checkpoints_path, f"best_metric_model.pth"))
                        torch.save({
                            'best_metric_epoch': best_metric_epoch,
                            'best_metric': best_metric,
                        },
                            os.path.join(checkpoints_path, f"best_metric_info.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

    print("Finished training!")


if __name__ == "__main__":
    sets = parse_opts()

    # parse the task
    cv_name = sets.cv5_name.split("_")[0]
    if cv_name == "poly":
        train_dict, val_dict, test_dict, train_transforms, val_transforms = prepare_my_data_poly(
            sets)
    else:
        raise NotImplementedError

    # load the dataset
    train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_num=len(train_dict), cache_rate=1.0,
                            num_workers=0)
    train_loader = DataLoader(
        train_ds, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers)

    if sets.val:
        val_ds = CacheDataset(
            data=val_dict, transform=val_transforms, cache_num=len(val_dict))
        val_loader = DataLoader(val_ds, batch_size=1,
                                shuffle=False, num_workers=sets.num_workers)
        data_loader = [train_loader, val_loader]
    else:
        data_loader = [train_loader, None]

    # getting model
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(sets.manual_seed)
    device, cpu_device = torch.device(f"cuda:{sets.gpu_id}"), torch.device("cpu")

    if sets.model == "dilatemobile":
        model = DilateMobileVitUnet(
            sets=sets, spatial_dims=2, in_channels=sets.n_input_channels, out_channels=sets.n_seg_classes).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        raise NotImplementedError
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=100, max_epochs=int(sets.n_epochs))
    dice_metric_train = DiceMetric(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # training
    trained_path = sets.save_folder
    tensorboard_log = os.path.join(trained_path, "tb")
    checkpoints_path = os.path.join(trained_path, "checkpoints")
    saved_path = [trained_path, tensorboard_log, checkpoints_path]
    for path in saved_path:
        if not os.path.exists(path):
            os.makedirs(path)
    writer = SummaryWriter(tensorboard_log)

    # train from resume
    current_epoch = 0
    best_metric_epoch = -1
    best_metric = -1
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sets.resume_path, checkpoint['epoch']))
            current_epoch = checkpoint['epoch']
        if os.path.exists(os.path.join(checkpoints_path, f"best_metric_info.pth")):
            checkpoint_best_info = torch.load(os.path.join(
                checkpoints_path, f"best_metric_info.pth"))
            best_metric_epoch = checkpoint_best_info['best_metric_epoch']
            best_metric = checkpoint_best_info['best_metric']

    train(sets, data_loader, model, current_epoch, writer,
          checkpoints_path, best_metric_epoch, best_metric)
