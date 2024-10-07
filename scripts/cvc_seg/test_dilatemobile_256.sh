#!/bin/bash


python test.py \
    --root_dir ./PolyResearch/CVC_ClinicDB/ \
    --cv5_name poly \
    --model dilatemobile \
    --model_type s \
    --fold 0 \
    --phase test \
    --n_input_channels 3 \
    --n_seg_classes 2 \
    --b_min 0.0 \
    --b_max 1.0 \
    --patch_size 256 256 \
    --resume_path ./checkpoints/best_metric_model.pth