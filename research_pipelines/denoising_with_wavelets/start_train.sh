#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/:../../third_party/Uformer/

python3 pytorch_wavelet_train.py \
    --model "resnet10t" \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train_SIDD_only/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 150 \
    --lr_milestones 2 \
    --image_size 256 \
    --batch_size 64 \
    --grad_accum_steps 1 \
    --visdom 9001 \
    --njobs 12 \
    --exp /media/alexey/SSDData/experiments/denoising/wavelets_paper/resnet10_v2/ \
    --preload_datasets \
    # --load /media/alexey/SSDData/experiments/denoising/unet_resnet10_v2/checkpoints/last.trh
