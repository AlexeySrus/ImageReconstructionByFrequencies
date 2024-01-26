#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 250 \
    --lr_milestones 1 \
    --image_size 256 \
    --batch_size 16 \
    --grad_accum_steps 4 \
    --visdom 9002 \
    --njobs 4 \
    --exp /media/alexey/SSDData/experiments/denoising/denoising_unet_fftcasplitfeats_v3/ \
    --preload_datasets \
    --load "/media/alexey/SSDData/experiments/denoising/denoising_unet_fftcasplitfeats_v3/checkpoints/last.trh"
