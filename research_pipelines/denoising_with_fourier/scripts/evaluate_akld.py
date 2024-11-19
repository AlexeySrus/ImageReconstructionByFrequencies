from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
import torch


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-a', '--a', type=str, required=True,
        help='Path to folder witht train/val subfolders'
    )
    parser.add_argument(
        '-b', '--b', type=str, required=True,
        help='Path to folder witht train/val subfolders'
    )
    return parser.parse_args()


def read_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img is not None, image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    res_tensor = res_tensor.permute(2, 0, 1).unsqueeze(0)
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


if __name__ == '__main__':
    args = parse_args()

    first_clear_folder = os.path.join(args.a, 'clear/')
    first_noisy_folder = os.path.join(args.a, 'noisy/')

    second_clear_folder = os.path.join(args.b, 'clear/')
    second_noisy_folder = os.path.join(args.b, 'noisy/')
    

    images = os.listdir(first_clear_folder)

    kdls = []

    for image_name in tqdm(images):
        clear_1 = read_image(os.path.join(first_clear_folder, image_name))
        noisy_1 = read_image(os.path.join(first_noisy_folder, image_name))

        clear_2 = read_image(os.path.join(second_clear_folder, image_name))
        noisy_2 = read_image(os.path.join(second_noisy_folder, image_name))

        sigma_real = estimate_sigma_gauss(noisy_1, clear_1)
        sigma_fake = estimate_sigma_gauss(noisy_2, clear_2)
        kl_dis = kl_gauss_zero_center(sigma_fake, sigma_real)

        kdls.append(kl_dis)

        
    print('Result KDL -- mean: {:.2f}, std: {:.2f}'.format(np.array(kdls).mean(), np.array(kdls).std()))
