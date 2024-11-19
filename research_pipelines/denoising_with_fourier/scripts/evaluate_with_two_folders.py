from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '--truth', type=str, required=True,
        help='Path to folder with gt images'
    )
    parser.add_argument(
        '--pred', type=str, required=True,
        help='Path to folder with predicted images'
    )
    return parser.parse_args()


def read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img is not None, image_path
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    args = parse_args()

    gt_folder = args.truth
    pred_folder = args.pred

    images = os.listdir(gt_folder)

    ssim_values = []
    psnr_values = []

    for image_name in tqdm(images):
        truth_img = read_image(os.path.join(gt_folder, image_name))
        pred_img = read_image(os.path.join(pred_folder, image_name))

        ssim_values.append(ssim(pred_img, truth_img, channel_axis=2))
        psnr_values.append(cv2.PSNR(pred_img, truth_img))

    print('Result PSNR -- mean: {:.2f}, std: {:.2f}'.format(np.array(psnr_values).mean(), np.array(psnr_values).std()))
    print('Result SSIM -- mean: {:.3f}, std: {:.3f}'.format(np.array(ssim_values).mean(), np.array(ssim_values).std()))
