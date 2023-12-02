#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_wavelet_train.py \
    --model "resnet10t" \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 50 \
    --lr_milestones 0 \
    --image_size 256 \
    --batch_size 32 \
    --visdom 9001 \
    --njobs 12 \
    --exp /media/alexey/SSDData/experiments/denoising/resnet10t_v2/ \
    --preload_datasets --no_load_optim \
    --load /media/alexey/SSDData/experiments/denoising/resnet10t_v2/checkpoints/best.trh
