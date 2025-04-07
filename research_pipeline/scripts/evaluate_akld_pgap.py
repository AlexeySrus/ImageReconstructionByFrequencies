from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import os
import torch
from torchmetrics.image import PeakSignalNoiseRatio as TorchPSNR


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Evaluate AKLD and PGap metrics')
    parser.add_argument(
        '-f', '--folder', type=str, required=True,
        help='Path to folder with train/val subfolders'
    )
    parser.add_argument(
        '--noisy_generated_images', type=str, required=True,
        help='Path to folder with noised frames'
    )
    return parser.parse_args()


def read_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img is not None, image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    res_tensor = res_tensor.permute(2, 0, 1)
    return res_tensor


def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)


def gaussblur(x, kernel, p=5, chn=3):
    x_pad = torch.nn.functional.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = torch.nn.functional.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y


def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance


def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=3).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=3)
    sigma.clamp_(min=1e-10)

    return sigma


def psnr_gap(img_original, img_restored1, img_restored2, psnr_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Вычисляет PSNR Gap между двумя восстановленными изображениями относительно исходного.
    
    Args:
        img_original (torch.Tensor): Исходное изображение.
        img_restored1 (torch.Tensor): Первое восстановленное изображение.
        img_restored2 (torch.Tensor): Второе восстановленное изображение.
        img_restored2 (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Функция для вычисления PSNR.
    
    Returns:
        float: Значение PSNR Gap.
    """
    psnr1 = psnr_function(img_original, img_restored1)
    psnr2 = psnr_function(img_original, img_restored2)
    return torch.abs(psnr1 - psnr2)


if __name__ == '__main__':
    args = parse_args()

    clear_folder = os.path.join(args.folder, 'clear/')
    noisy_folder = os.path.join(args.folder, 'noisy/')

    images = os.listdir(clear_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    psnr_measure = TorchPSNR(data_range=1.0).to(device)

    generated_folder = args.noisy_generated_images

    kdls = []
    pgaps = []

    for image_name in tqdm(images):
        clear_gt = read_image(os.path.join(clear_folder, image_name)).to(device)
        noisy_gt = read_image(os.path.join(noisy_folder, image_name)).to(device)
        noisy_pred = read_image(os.path.join(generated_folder, image_name)).to(device)

        sigma_real = estimate_sigma_gauss(noisy_gt, clear_gt)
        sigma_fake = estimate_sigma_gauss(noisy_pred, clear_gt)

        kl_dis = kl_gauss_zero_center(sigma_fake, sigma_real)

        kdls.append(kl_dis.item())

        pgap_value = psnr_gap(
            img_original=clear_gt,
            img_restored1=noisy_gt,
            img_restored2=noisy_pred,
            psnr_function=psnr_measure
        )
        pgaps.append(pgap_value.item())

    print('Result KDL -- mean: {:.3f}, std: {:.3f}'.format(np.array(kdls).mean(), np.array(kdls).std()))
    print('Result PGap -- mean: {:.2f}, std: {:.2f}'.format(np.array(pgaps).mean(), np.array(pgaps).std()))
