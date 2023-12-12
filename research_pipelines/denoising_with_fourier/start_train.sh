#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 250 \
    --lr_milestones 2 \
    --image_size 256 \
    --batch_size 16 \
    --grad_accum_steps 1 \
    --visdom 9001 \
    --njobs 4 \
    --exp /media/alexey/SSDData/experiments/denoising/fftcnn_fftcomplexattention_exp/ \
    --preload_datasets \
    --load /media/alexey/SSDData/experiments/denoising/fftcnn_fftcomplexattention_exp/checkpoints/last.trh
