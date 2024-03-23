from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import os
import sys

CURRENT_PATH = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(CURRENT_PATH, '../'))

from utils.image_utils import pad_image_to_inference


def cv_add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    std = sigma
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image.copy().astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with training/val set'
    )
    parser.add_argument(
        '--grayscale', action='store_true',
        help='Use grayscale images to evaluate'
    )
    parser.add_argument(
        '--minsize', type=int, required=False, default=256,
        help='Minimum image size (default: 256)'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    output_clear_images_folder = os.path.join(args.output, 'clear/')
    output_noisy_images_folder = os.path.join(args.output, 'noisy/')

    os.makedirs(output_clear_images_folder, exist_ok=True)
    os.makedirs(output_noisy_images_folder, exist_ok=True)

    noisy_sigmas = [15, 25, 50, 75, 150]

    for img_name in tqdm(os.listdir(args.input)):
        bname, ext = os.path.splitext(img_name)

        input_path = os.path.join(args.input, img_name)

        img = cv2.imread(
            input_path,
            cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
        )

        if min(img.shape[:2]) < args.minsize:
            img = pad_image_to_inference(img)

        for k in range(len(noisy_sigmas)):
            s1 = noisy_sigmas[k - 1] if k > 0 else 1
            s2 = noisy_sigmas[k]

            sigma = np.random.randint(s1, s2 + 1)

            nimg = cv_add_gaussian_noise(img, sigma)

            image_basename = '{}_{}-{}'.format(
                bname, s1, s2
            )
            
            clear_path = os.path.join(
                output_clear_images_folder,
                image_basename + ext
            )
            noisy_path = os.path.join(
                output_noisy_images_folder,
                image_basename + '.png'
            )

            copyfile(input_path, clear_path)
            is_save = cv2.imwrite(
                noisy_path,
                nimg,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            assert is_save, noisy_path
