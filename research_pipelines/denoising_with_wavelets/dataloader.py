from typing import Tuple, Optional, List, Union
import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from utils.image_utils import random_crop_with_transforms, pil_load_image as load_image
from utils.tensor_utils import preprocess_image


class PairedDenoiseDataset(Dataset):
    def __init__(self,
                 noisy_images_path,
                 clear_images_path,
                 need_crop: bool = False,
                 window_size: int = 224,
                 optional_dataset_size: Optional[int] = None,
                 preload: bool = False,
                 return_names: bool = False):
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
        self.return_names = return_names

        self.names = [img_name for img_name in os.listdir(clear_images_path)]

        if preload:
            print('Loading images into RAM:')
            for key in tqdm(self.images_keys):
                self.noisy_images[key] = load_image(self.noisy_images[key])
                self.clear_images[key] = load_image(self.clear_images[key])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, str]]:
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

        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2YCrCb)
        clear_image = cv2.cvtColor(clear_image, cv2.COLOR_RGB2YCrCb)

        if self.return_names:
            img_name = self.names[idx]
            return preprocess_image(noisy_image, 0, 1), preprocess_image(clear_image, 0, 1), img_name

        return preprocess_image(noisy_image, 0, 1), preprocess_image(clear_image, 0, 1)


class SyntheticNoiseDataset(Dataset):
    def __init__(self, 
                 clear_images_path, 
                 window_size: int = 224,
                 optional_dataset_size: Optional[int] = None,
                 preload: bool = False):
        self.clear_images = [
            os.path.join(clear_images_path, img_name)
            for img_name in os.listdir(clear_images_path)
        ]

        if preload:
            print('Loading images into RAM:')
            self.clear_images = [
                load_image(imgp)
                for imgp in tqdm(self.clear_images)
            ]

        self.dataset_size = len(self.clear_images) if optional_dataset_size is None else optional_dataset_size
        self.window_size = window_size

        self.noise_transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 150.0), always_apply=True),
                A.ISONoise(always_apply=True),
                A.MultiplicativeNoise(always_apply=True)
            ], p=0.8)
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = _idx % len(self.clear_images)

        if np.random.randint(1, 101) > 80:
            rand_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            clear_image = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            clear_image[:, :] = rand_color
        else:
            clear_image = self.clear_images[idx]

        if isinstance(clear_image, str):
            clear_image = load_image(clear_image)

        assert min(clear_image.shape[:2]) >= self.window_size

        clear_crop = random_crop_with_transforms(
            clear_image, None,
            window_size=self.window_size,
            random_swap=False
        )

        if np.random.randint(1, 101) > 20:
            noisy_crop = self.noise_transform(image=clear_crop)['image']
        else:
            noisy_crop = clear_crop.copy()

        noisy_crop = cv2.cvtColor(noisy_crop, cv2.COLOR_RGB2YCrCb)
        clear_crop = cv2.cvtColor(clear_crop, cv2.COLOR_RGB2YCrCb)

        return preprocess_image(noisy_crop, 0, 1), preprocess_image(clear_crop, 0, 1)


if __name__ == '__main__':
    val_data = (
        '/media/alexey/SSDData/datasets/denoising_dataset/val/noisy/',
        '/media/alexey/SSDData/datasets/denoising_dataset/val/clear/'
    )

    dataset = SyntheticNoiseDataset(val_data[1])
    for i in range(len(dataset)):
        n, c = dataset[2]
        print(i, n.shape, c.shape)
