#!/bin/bash

export PYTHONPATH=./:../../third_party/pytorch-attention/

python3 pytorch_fourier_train.py \
    --train_data_folder /media/alexey/SSDData/datasets/denoising_dataset/cyclegan_noisy/sidd_with_synth_lsdir/ \
    --validation_data_folder /media/alexey/SSDData/datasets/denoising_dataset/val/ \
    --epochs 150 \
    --lr 0.0001 \
    --lr_milestones 0 \
    --substracted_noise \
    --architecture 'nafnet' \
    --attention_mode 'none' \
    --interpolation_mode 'bilinear' \
    --image_size 256 \
    --batch_size 16 \
    --grad_accum_steps 1 \
    --visdom 9001 \
    --njobs 8 \
    --exp /media/alexey/SSDData/experiments/denoising/generation_paper/NAFNET_synth_with_SIDD2s/ \
    --preload_datasets \
    # --load "/media/alexey/SSDData/experiments/denoising/generation_paper/DNCNN_synth_with_SIDD/checkpoints/last.trh" \
    # --no_load_optim
