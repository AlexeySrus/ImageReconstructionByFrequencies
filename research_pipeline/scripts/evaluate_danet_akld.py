from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import os
import torch
import sys


CURRENT_PATH = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(CURRENT_PATH, '../'))
sys.path.insert(0, os.path.join(CURRENT_PATH, '../../../third_party/DANet/'))


from utils.common_window_inference import denoise_inference
from networks import UNetG, sample_generator
from uformer import get_base_uformer_model


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-f', '--folder', type=str, required=True,
        help='Path to folder with train/val subfolders'
    )
    parser.add_argument(
        '--noise_generator_weights', type=str, required=True,
        help='Path to file of DANet Generator weights'
    )
    parser.add_argument(
        '-k', '--k', type=int, required=False, default=15,
        help='Count of generations to estimave KLD'
    )
    return parser.parse_args()


def read_image(image_path: str) -> torch.Tensor:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img is not None, image_path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    res_tensor = res_tensor.permute(2, 0, 1)
    return res_tensor


def danet_inference(inp_tensor: torch.Tensor, net_g: torch.nn.Module) -> torch.Tensor:
    out = sample_generator(net_g, inp_tensor)
    out = torch.clamp(out, 0.0, 1.0)
    return out


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

    clear_folder = os.path.join(args.folder, 'clear/')
    noisy_folder = os.path.join(args.folder, 'noisy/')

    images = os.listdir(clear_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    net = UNetG(3, wf=32, depth=5).to(device)
    # net = get_base_uformer_model(image_size=256, in_ch=4, out_ch=3).to(device)
    net.load_state_dict(torch.load(args.noise_generator_weights, map_location=device)['G'])
    _ = net.eval()

    kdls = []

    for image_name in tqdm(images):
        clear_gt = read_image(os.path.join(clear_folder, image_name)).to(device)
        noisy_gt = read_image(os.path.join(noisy_folder, image_name)).to(device)

        sigma_real = estimate_sigma_gauss(noisy_gt, clear_gt)

        for _ in range(args.k):
            inference_func = lambda tx: danet_inference(tx, net)
            with torch.no_grad():
                nimg = denoise_inference(
                    clear_gt,
                    inference_func,
                    256,
                    16,
                    crop_size=16
                )
            
            nimg = nimg[0]
            sigma_fake = estimate_sigma_gauss(nimg, clear_gt)

            kl_dis = kl_gauss_zero_center(sigma_fake, sigma_real)

            kdls.append(kl_dis.item())

        
    print('Result KDL -- mean: {:.2f}, std: {:.2f}'.format(np.array(kdls).mean(), np.array(kdls).std()))
