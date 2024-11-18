#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --validation_data_folder /media/alexey/HDDData/datasets/image_denoising/MRI/val_dataset/ \
    --synthetic_data_paths /media/alexey/HDDData/datasets/image_denoising/MRI/images/ \
    --epochs 250 \
    --lr 0.0001 \
    --lr_milestones 1 \
    --image_size 256 \
    --batch_size 4 \
    --grad_accum_steps 8 \
    --architecture 'uformer' \
    --visdom 9001 \
    --njobs 8 \
    --exp /media/alexey/SSDData/experiments/denoising/fft_attention_paper/mri_uformer_with_fswaunet/ \
    --preload_datasets \
    --use_grayscale \
    --load "/media/alexey/SSDData/experiments/denoising/fft_attention_paper/mri_uformer_with_fswaunet/checkpoints/last.trh" \
    # --no_load_optim
