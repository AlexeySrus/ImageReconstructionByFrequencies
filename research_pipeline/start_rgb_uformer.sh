#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/train/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --synthetic_data_paths /media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/ \
    --epochs 50 \
    --lr 0.001 \
    --lr_milestones -1000 \
    --substracted_noise \
    --attention_mode 'none' \
    --architecture 'uformer' \
    --interpolation_mode 'lanczos4' \
    --image_size 256 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --visdom 9100 \
    --njobs 8 \
    --exp /media/alexey/SSDData/experiments/denoising/main_uformer_denoiser/ \
    --preload_datasets \
    --load "/media/alexey/SSDData/experiments/denoising/main_uformer_denoiser/checkpoints/last.trh" \
    # --no_load_optim
