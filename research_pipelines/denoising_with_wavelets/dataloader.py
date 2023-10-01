from typing import Tuple, Optional, List
import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pywt
import random
import logging

from utils.image_utils import random_crop_with_transforms, load_image
from utils.tensor_utils import preprocess_image


class SeriesAndClearDataset(Dataset):
    def __init__(self,
                 series_folders_path,
                 clear_images_path,
                 window_size,
                 dataset_size):
        self.series_folders_pathes = [
            os.path.join(series_folders_path, sfp)
            for sfp in os.listdir(series_folders_path)
        ]

        self.clear_images_pathes = [
            os.path.join(clear_images_path, cip)
            for cip in os.listdir(clear_images_path)
        ]

        assert len(self.series_folders_pathes) == len(
            self.series_folders_pathes)

        self.sort_key = lambda s: int(s.split('_')[-1].split('.')[0])

        self.series_folders_pathes.sort(key=self.sort_key)
        self.clear_images_pathes.sort(key=self.sort_key)

        self.series_folders_pathes = [
            [os.path.join(sfp, img_name) for img_name in os.listdir(sfp)]
            for sfp in self.series_folders_pathes
        ]

        self.window_size = window_size
        self.dataset_size = dataset_size

    def get_random_images(self):
        select_series_index = random.randint(
            0,
            len(self.series_folders_pathes) - 1
        )

        select_image_index = random.randint(
            0,
            len(self.series_folders_pathes[select_series_index]) - 1
        )

        select_image = load_image(
            self.series_folders_pathes[select_series_index][select_image_index]
        )

        clear_image = load_image(self.clear_images_pathes[select_series_index])

        return select_image, clear_image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.get_random_images(), self.window_size
        )

        return preprocess_image(crop1, 0, 1), preprocess_image(crop2, 0, 1)


class SeriesAndComputingClearDataset(Dataset):
    def __init__(self,
                 images_series_folder,
                 clear_images_path,
                 window_size,
                 dataset_size):
        self.series_folders_pathes = [
            os.path.join(images_series_folder, sfp)
            for sfp in os.listdir(images_series_folder)
        ]

        self.clear_series_pathes = [
            os.path.join(clear_images_path, cfp)
            for cfp in os.listdir(clear_images_path)
        ]

        assert len(self.series_folders_pathes) == len(
            self.clear_series_pathes)

        self.sort_key = lambda s: int(s.split('_')[-1].split('.')[0])

        self.series_folders_pathes.sort(key=self.sort_key)
        self.clear_series_pathes.sort(key=self.sort_key)

        self.series_folders_pathes = [
            [os.path.join(sfp, img_name) for img_name in os.listdir(sfp)]
            for sfp in self.series_folders_pathes
        ]

        self.clear_series_pathes = [
            [os.path.join(cfp, img_name) for img_name in os.listdir(cfp)]
            for cfp in self.clear_series_pathes
        ]

        for i in range(len(self.series_folders_pathes)):
            self.series_folders_pathes[i].sort(key=self.sort_key)
            self.clear_series_pathes[i].sort(key=self.sort_key)

        self.series_folders_images = []
        self.clear_series_folders_images = []

        print('Loading images into RAM:')
        for i in tqdm(range(len(self.series_folders_pathes))):
            self.series_folders_images.append(
                [
                    load_image(self.series_folders_pathes[i][j])
                    for j in range(len(self.series_folders_pathes[i]))
                ]
            )

            self.clear_series_folders_images.append(
                [
                    load_image(self.clear_series_pathes[i][j])
                    for j in range(len(self.clear_series_pathes[i]))
                ]
            )

        self.window_size = window_size
        self.dataset_size = dataset_size

    def get_random_images(self, idx: Optional[int] = None):
        select_series_index = np.random.randint(
            0,
            len(self.series_folders_pathes)
        )

        select_image_index = np.random.randint(
            0,
            len(self.series_folders_pathes[select_series_index])
        )

        if isinstance(self.series_folders_images[select_series_index][select_image_index], str):
            select_image = load_image(self.series_folders_images[
                select_series_index][select_image_index])
        else:
            select_image = self.series_folders_images[select_series_index][select_image_index]

        if isinstance(self.clear_series_folders_images[select_series_index][select_image_index], str):
            clear_image = load_image(self.clear_series_folders_images[
                select_series_index][select_image_index])
        else:
            clear_image = self.clear_series_folders_images[select_series_index][select_image_index]

        return select_image, clear_image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        crop1, crop2 = random_crop_with_transforms(
            *self.get_random_images(),
            window_size=self.window_size,
            random_swap=False
        )

        return preprocess_image(crop1, 0, 1), preprocess_image(crop2, 0, 1)


class PairedDenoiseDataset(Dataset):
    def __init__(self,
                 noisy_images_path,
                 clear_images_path,
                 need_crop: bool = False,
                 window_size: int = 224,
                 optional_dataset_size: Optional[int] = None,
                 preload: bool = False):
        self.noisy_images = {
            os.path.splitext(img_name)[0]: os.path.join(noisy_images_path, img_name)
            for img_name in os.listdir(noisy_images_path)
        }
        self.clear_images = {
            os.path.splitext(img_name)[0]: os.path.join(clear_images_path, img_name)
            for img_name in os.listdir(clear_images_path)
        }

        assert set(self.noisy_images.keys()) == set(self.clear_images.keys())

        self.images_keys = list(self.noisy_images.keys())
        self.dataset_size = len(self.images_keys) if optional_dataset_size is None else optional_dataset_size
        self.window_size = window_size
        self.need_crop = need_crop

        if preload:
            print('Loading images into RAM:')
            for key in tqdm(self.images_keys):
                self.noisy_images[key] = load_image(self.noisy_images[key])
                self.clear_images[key] = load_image(self.clear_images[key])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _idx) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = _idx % len(self.images_keys)

        noisy_image = self.noisy_images[self.images_keys[idx]]
        clear_image = self.clear_images[self.images_keys[idx]]

        if isinstance(noisy_image, str):
            noisy_image = load_image(noisy_image)
        if isinstance(clear_image, str):
            clear_image = load_image(clear_image)

        if self.need_crop:
            noisy_image, clear_image = random_crop_with_transforms(
                noisy_image, clear_image,
                window_size=self.window_size,
                random_swap=False
            )

        return preprocess_image(noisy_image, 0, 1), preprocess_image(clear_image, 0, 1)


class SyntheticNoiseDataset(Dataset):
    def __init__(self, clear_images_path, window_size: int = 224):
        self.clear_images = [
            os.path.join(clear_images_path, img_name)
            for img_name in os.listdir(clear_images_path)
        ]

        self.dataset_size = len(self.clear_images)
        self.window_size = window_size

        self.noise_transform = A.Compose([
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(var_limit=(10.0, 150.0)),
                A.ISONoise(),
                A.MultiplicativeNoise()
            ], p=1.0)
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        clear_image = load_image(self.clear_images[idx])

        if min(clear_image.shape[:2]) <= self.window_size:
            clear_image = cv2.resize(
                clear_image,
                (self.window_size + 2, self.window_size + 2),
                interpolation=cv2.INTER_AREA
            )

        noisy_image = self.noise_transform(image=clear_image)['image']

        clear_crop, noisy_crop = random_crop_with_transforms(
            clear_image, noisy_image,
            window_size=self.window_size,
            random_swap=False
        )

        return preprocess_image(noisy_crop, 0, 1), preprocess_image(clear_crop, 0, 1)


if __name__ == '__main__':
    val_data = (
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/noisy/',
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/clear/'
    )

    dataset = SyntheticNoiseDataset(val_data[1])
    for i in range(len(dataset)):
        n, c = dataset[2]
        print(i, n.shape, c.shape)