from typing import Tuple, Optional, List, Union, Dict
import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
from tqdm import tqdm

from utils.image_utils import random_crop_with_transforms, pil_load_image as load_image
from utils.image_utils import generate_additive_gaussian_noise, generate_additive_poisson_noise
from utils.fft_mask_utils import ssdu_masks
from utils.tensor_utils import preprocess_image


MRI_CONGIF: Dict[str, int] = {
    'RAND_FILL': 97,
    'ADD_NOISE': 10,
    'GAUSS_AND_POISSON_NOISE': 20,
    'POISSON_NOISE': 80,
    'FFT_MASK': 50,
    'MIXUP': 95
}
RGB_CONFIG: Dict[str, int] = {
    'RAND_FILL': 95,
    'ADD_NOISE': 10,
    'GAUSS_AND_POISSON_NOISE': 40,
    'POISSON_NOISE': 80,
    'FFT_MASK': 50,
    'MIXUP': 85
}

SYNTH_CONFIG: Dict[str, int] = RGB_CONFIG


def cv_convert_to_rgb_or_grayscale(image: np.ndarray, to_ycrcb: bool, to_grayscale: bool):
    if to_ycrcb and not to_grayscale:
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif to_grayscale:
        res_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        res_img = np.expand_dims(res_img, axis=2)
        return res_img
    return image


def get_random_value_from_interval(a: float, b: float) -> float:
    assert b > a
    return a + np.random.rand() * (b - a)


class PairedDenoiseDataset(Dataset):
    def __init__(self,
                 noisy_images_path,
                 clear_images_path,
                 need_crop: bool = False,
                 window_size: int = 224,
                 optional_dataset_size: Optional[int] = None,
                 preload: bool = False,
                 return_names: bool = False,
                 use_ycrcb: bool = False,
                 grayscale: bool = False):
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
        self.use_ycrcb = use_ycrcb
        self.grayscale = grayscale

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

        if self.return_names:
            img_name = self.names[idx]
            return preprocess_image(
                noisy_image, 0, 1, self.use_ycrcb, self.grayscale), preprocess_image(
                    clear_image, 0, 1, self.use_ycrcb, self.grayscale), img_name

        return preprocess_image(
            noisy_image, 0, 1, self.use_ycrcb, self.grayscale), preprocess_image(
                clear_image, 0, 1, self.use_ycrcb, self.grayscale)


class SyntheticNoiseDataset(Dataset):
    support_mask_kernels = [4, 8, 16, 32, 64]
    interpolations = [
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        # cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4
    ]

    def __init__(self, 
                 clear_images_path, 
                 window_size: int = 224,
                 optional_dataset_size: Optional[int] = None,
                 preload: bool = False,
                 use_ycrcb: bool = False,
                 grayscale: bool = False):
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
        self.use_ycrcb = use_ycrcb
        self.grayscale = grayscale

        self.noise_transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 150.0), always_apply=True),
                A.ISONoise(always_apply=True),
                A.MultiplicativeNoise(always_apply=True)
            ], p=0.5)
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = _idx % len(self.clear_images)

        if np.random.randint(1, 101) > SYNTH_CONFIG['RAND_FILL']:
            rand_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            clear_image = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            clear_image[:, :] = rand_color
        else:
            clear_image = self.clear_images[idx]

        if isinstance(clear_image, str):
            clear_image = load_image(clear_image)

        assert min(clear_image.shape[:2]) >= self.window_size

        if self.grayscale and min(clear_image.shape[:2]) > 256:
            min_scale = self.window_size / min(clear_image.shape[:2])
            scale = get_random_value_from_interval(min_scale, 1.0) + 1E-5

            clear_image = cv2.resize(
                clear_image, None, fx=scale, fy=scale, 
                interpolation=np.random.choice(self.interpolations)
            )

        clear_crop = random_crop_with_transforms(
            clear_image, None,
            window_size=self.window_size,
            random_swap=False
        )

        if np.random.randint(1, 101) > SYNTH_CONFIG['ADD_NOISE']:
            if np.random.randint(1, 101) > SYNTH_CONFIG['GAUSS_AND_POISSON_NOISE']:
                if np.random.randint(1, 101) > SYNTH_CONFIG['POISSON_NOISE']:
                    noisy_crop = generate_additive_poisson_noise(clear_crop)
                else:
                    std = np.random.uniform(1, 90)
                    use_fft_noise = self.grayscale and np.random.choice([False, False, False, True])
                    noisy_crop = generate_additive_gaussian_noise(clear_crop, std, use_fft_noise)
            else:
                noisy_crop = self.noise_transform(image=clear_crop)['image']

            noisy_crop = np.clip(noisy_crop, 0.0, 255.0).astype(np.uint8)
        else:
            noisy_crop = clear_crop.copy()

        if self.grayscale and np.random.randint(1, 101) > SYNTH_CONFIG['FFT_MASK']:
            block_size = np.random.choice(self.support_mask_kernels)
            rho = get_random_value_from_interval(0.05, 0.4)

            noisy_crop = ssdu_masks(
                rho=rho,
                small_acs_block=(block_size, block_size)
            ).apply_fft_matrix(noisy_crop)

        return preprocess_image(
            noisy_crop, 0, 1, self.use_ycrcb, self.grayscale), preprocess_image(
                clear_crop, 0, 1, self.use_ycrcb, self.grayscale)


if __name__ == '__main__':
    val_data = (
        '/media/alexey/SSDData/datasets/denoising_dataset/val/noisy/',
        '/media/alexey/SSDData/datasets/denoising_dataset/val/clear/'
    )

    dataset = SyntheticNoiseDataset(val_data[1])
    for i in range(len(dataset)):
        n, c = dataset[2]
        print(i, n.shape, c.shape)
