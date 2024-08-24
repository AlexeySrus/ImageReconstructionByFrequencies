#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 50 \
    --lr 0.0001 \
    --lr_milestones 1 \
    --substracted_noise \
    --attention_mode 'full' \
    --interpolation_mode 'lanczos4' \
    --image_size 256 \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --visdom 9001 \
    --njobs 8 \
    --exp /media/alexey/SSDData/experiments/denoising/dissertation/all_data/lanczos4/unet_full/ \
    --preload_datasets \
    --load "/media/alexey/SSDData/experiments/denoising/dissertation/all_data/lanczos4/unet_full/checkpoints/last.trh" \
    # --no_load_optim
