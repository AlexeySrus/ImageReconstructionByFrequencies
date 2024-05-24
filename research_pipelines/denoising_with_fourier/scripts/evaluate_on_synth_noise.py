from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
import os

CURRENT_PATH = os.path.dirname(__file__)

from FFTCNN.combined_attn_unet import FFTAttentionUNet as FFTCNN
from utils.window_inference import eval_denoise_inference


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
   
    parser.add_argument(
        '-f', '--folder', type=str, required=True,
        help='Path to folder with images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with output visualizations (optional)'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable printing metrics per each sample)'
    )
    parser.add_argument(
        '--y-channel-only', action='store_true',
        help='Evaluate only on Y component from YCrCb color space'
    )
    parser.add_argument(
        '--use_tta', action='store_true',
        help='Use test time augmentations until inference'
    )
    parser.add_argument(
        '--attention_mode', type=str, required=False, default='full',
        choices=['full', 'ca', 'sa', 'cbam', 'none'],
        help='Attention mode from \'full\', \'ca\', \'sa\', \'cbam\', \'none\'.'
    )
    return parser.parse_args()


def add_gaussian_noise(image: np.ndarray, std: int = 15) -> np.ndarray:
    assert std > 0
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0.0, 255.0).astype(np.uint8)
    return noisy_image


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


if __name__ == '__main__':
    args = parse_args()

    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FFTCNN(use_substraction=True, attention_mode=args.attention_mode).to(device)

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)
    model.load_state_dict(load_data['model'])
    model.eval()

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    psnr_values = []
    ssim_values = []

    if args.output is None:
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/eval_results_SYNTH/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    clear_folder = args.folder

    noisy_sigmas = [5, 10, 15, 25, 30, 50]

    for noise_sigma in noisy_sigmas:
        print('SIGMA VALUE: {}'.format(noise_sigma))

        output_save_folder = os.path.join(output_folder, 'sigma_{}'.format(noise_sigma))
        os.makedirs(output_save_folder, exist_ok=True)

        dataset_wavelet_losses: List[List[Dict[str, float]]] = []

        for image_name in tqdm(os.listdir(clear_folder)):
            gt_image_path = os.path.join(clear_folder, image_name)
            gt_img = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)
            img = add_gaussian_noise(gt_img, noise_sigma)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

            assert img.shape[0] == gt_img.shape[0] and img.shape[1] == gt_img.shape[1], image_name
            input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)

            with torch.no_grad():
                restored_image = eval_denoise_inference(
                    tensor_img=input_tensor, model=model, window_size=imgsz, 
                    batch_size=4, crop_size=imgsz // 32, use_tta=args.use_tta, device=device
                )

            input_tensor = input_tensor.to('cpu')
            pred_image = restored_image.to('cpu')

            pred_image = torch.clamp(pred_image, 0, 1)
            pred_image = tensor_to_image(pred_image)

            rgb_pred = pred_image
            rgb_gt = gt_img

            if args.y_channel_only:
                ssim_values.append(ssim(cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2YCrCb)[..., 0], cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2YCrCb)[..., 0]))
                psnr_values.append(cv2.PSNR(cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2YCrCb)[..., 0], cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2YCrCb)[..., 0]))
            else:
                ssim_values.append(ssim(rgb_pred, rgb_gt, channel_axis=2))
                psnr_values.append(cv2.PSNR(rgb_pred, rgb_gt))

            if args.verbose:
                print('Image: {}, PSNR: {:.2f}, SSIM: {:.3f}'.format(image_name, psnr_values[-1], ssim_values[-1]))

            cv2.imwrite(
                os.path.join(output_save_folder, image_name),
                cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
            )

        print('Result PSNR -- mean: {:.2f}, std: {:.2f}'.format(np.array(psnr_values).mean(), np.array(psnr_values).std()))
        print('Result SSIM -- mean: {:.3f}, std: {:.3f}'.format(np.array(ssim_values).mean(), np.array(ssim_values).std()))
