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
import logging

from utils.image_utils import random_crop_with_transforms, load_image, split_by_wavelets
from utils.tensor_utils import preprocess_image


class WaveletSuperSamplingDataset(Dataset):
    def __init__(self, folder_path, window_size: int = 224, dataset_size: int = 1000):
        images_names_list = os.listdir(folder_path)
        images_names_list.sort()

        self.images_paths = [
            os.path.join(folder_path, image_name)
            for image_name in images_names_list
        ]

        self.window_size = window_size
        self.dataset_size = dataset_size
        self.images_count = len(self.images_paths)

        self.interpolations = [
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            None
        ]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(0, self.images_count)
        image = load_image(self.images_paths[image_idx])

        if min(image.shape[:2]) < self.window_size:
            logging.info('Image {} so small, resizing!'.format(self.images_paths[image_idx]))
            image = cv2.resize(image, (self.window_size + 5, self.window_size + 5), interpolation=cv2.INTER_AREA)

        crop = random_crop_with_transforms(
            image1=image,
            window_size=self.window_size
        )

        selected_inter_method: Optional[int] = self.interpolations[np.random.randint(0, len(self.interpolations))]
        # TODO: Add transform which changed OpenCV image to LL wavelet representation
        selected_inter_method = None
        ycrcb_ll_crop: Optional[np.ndarray] = None

        if selected_inter_method is not None:
            lr_crop = cv2.resize(
                crop,
                (self.window_size // 2, self.window_size // 2),
                interpolation=selected_inter_method
            )
            ycrcb_ll_crop = cv2.cvtColor(lr_crop, cv2.COLOR_RGB2YCrCb)
            ycrcb_ll_crop = ycrcb_ll_crop.astype(np.float32) / 255.0 * self.window_size * 2

        ycrcb_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb_crop)

        # LL, LH, HL, HH <- C
        y_ll, y_lh, y_hl, y_hh = split_by_wavelets(y)
        cr_ll, cr_lh, cr_hl, cr_hh = split_by_wavelets(cr)
        cb_ll, cb_lh, cb_hl, cb_hh = split_by_wavelets(cb)

        if selected_inter_method is None:
            ycrcb_ll_crop = cv2.merge((y_ll, cr_ll, cb_ll))

        # 9 channels
        gt_wavelets = cv2.merge((y_lh, y_hl, y_hh, cr_lh, cr_hl, cr_hh, cb_lh, cb_hl, cb_hh))

        return preprocess_image(ycrcb_ll_crop), preprocess_image(gt_wavelets, 0, 1), preprocess_image(ycrcb_crop)


class SuperSamplingDataset(WaveletSuperSamplingDataset):
    def __init__(self, folder_path, window_size: int = 224, dataset_size: int = 1000):
        super().__init__(folder_path, window_size, dataset_size)

        self.interpolations = [
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC
        ]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_idx = np.random.randint(0, self.images_count)
        image = load_image(self.images_paths[image_idx])

        if min(image.shape[:2]) < self.window_size:
            logging.info('Image {} so small, resizing!'.format(self.images_paths[image_idx]))
            image = cv2.resize(image, (self.window_size + 5, self.window_size + 5), interpolation=cv2.INTER_AREA)

        crop = random_crop_with_transforms(
            image1=image,
            window_size=self.window_size
        )

        selected_inter_method: int = self.interpolations[np.random.randint(0, len(self.interpolations))]

        low_res_crop = cv2.resize(
            crop,
            (self.window_size // 2, self.window_size // 2),
            interpolation=selected_inter_method
        )

        return preprocess_image(low_res_crop, 0, 1), preprocess_image(crop, 0, 1)
