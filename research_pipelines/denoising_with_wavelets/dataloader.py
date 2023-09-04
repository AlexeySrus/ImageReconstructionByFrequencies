from typing import Tuple, Optional
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

        # print('Loading images into RAM:')
        for i in tqdm(range(len(self.series_folders_pathes))):
            self.series_folders_images.append(
                [
                    self.series_folders_pathes[i][j]    # load_image(self.series_folders_pathes[i][j])
                    for j in range(len(self.series_folders_pathes[i]))
                ]
            )

            self.clear_series_folders_images.append(
                [
                    self.clear_series_pathes[i][j]  # load_image(self.clear_series_pathes[i][j])
                    for j in range(len(self.clear_series_pathes[i]))
                ]
            )

        self.window_size = window_size
        self.dataset_size = dataset_size

    def get_random_images(self):
        select_series_index = np.random.randint(
            0,
            len(self.series_folders_pathes)
        )

        select_image_index = np.random.randint(
            0,
            len(self.series_folders_pathes[select_series_index])
        )

        select_image = load_image(self.series_folders_images[
            select_series_index][select_image_index])

        clear_image = load_image(self.clear_series_folders_images[
            select_series_index][select_image_index])

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