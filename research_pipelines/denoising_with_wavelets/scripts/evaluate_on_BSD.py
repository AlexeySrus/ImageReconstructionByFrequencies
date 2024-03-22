from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
import os

CURRENT_PATH = os.path.dirname(__file__)

from WTSNet.wts_timm import UnetTimm, WTSNetTimm, SharpnessHead, SharpnessHeadForYChannel
from utils.window_inference import eval_denoise_inference, denoise_inference
from utils.cas import contrast_adaptive_sharpening
from utils.haar_utils import HaarForward


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '-n', '--name', type=str, required=False, default='resnet10t',
        help='Timm model name'
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
        '--use_tta', action='store_true',
        help='Use test time augmentations until inference'
    )
    parser.add_argument(
        '--use_unet', action='store_true',
        help='Use classic U-Net network decoder architecture'
    )
    parser.add_argument(
        '--use_sharpness_head', action='store_true',
        help='Use sharpness head'
    )
    parser.add_argument(
        '-l', '--levels', type=int, required=False, default=5,
        help='Wavelte levels'
    )
    parser.add_argument(
        '--grayscale', action='store_true',
        help='Use grayscale images to evaluate'
    )
    return parser.parse_args()


def pad_image_to_inference(image: np.ndarray) -> np.ndarray:
    res = image.copy()

    nearest_size = 2 ** int(np.ceil(np.log2(max(image.shape[:2]))))

    res = cv2.copyMakeBorder(
        res, 
        0, nearest_size - image.shape[0], 
        0, nearest_size - image.shape[1], 
        cv2.BORDER_REFLECT
    )

    return res


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


class DWTHaar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = HaarForward()

    def forward(self, x):
        out = self.dwt(x)
        step = out.size(1) // 4
        ll = out[:, :step]
        lh = out[:, step:step*2]
        hl = out[:, step*2:step*3]
        hh = out[:, step*3:]
        return [ll, lh, hl, hh]


def l2_loss(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - truth) / pred.size(0)


def compute_wavelets_losses(pred: torch.Tensor, truth: torch.Tensor, 
                            wavelets_levels: int = 5, 
                            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.functional.smooth_l1_loss) -> List[Dict[str, float]]:
    dwt = DWTHaar()

    lvl_losses: List[Dict[str, float]] = []

    pred_ll = pred
    trurh_ll = truth

    for q in range(wavelets_levels):
        pred_ll, plh, phl, phh = dwt(pred_ll)
        trurh_ll, tlh, thl, thh = dwt(trurh_ll)

        lh_loss = loss_fn(plh, tlh).item()
        hl_loss = loss_fn(phl, thl).item()
        hh_loss = loss_fn(phh, thh).item()

        lvl_losses.append(
            {
                'LH': lh_loss,
                'HL': hl_loss,
                'HH': hh_loss
            }
        )

    return lvl_losses


if __name__ == '__main__':
    args = parse_args()

    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)

    if args.use_unet:
        try:
            model = UnetTimm(model_name=args.name, use_biliniar=False).to(device)
            model.load_state_dict(load_data['model'])
        except:
            model = UnetTimm(model_name=args.name, use_biliniar=True).to(device)
            model.load_state_dict(load_data['model'])
            print('Use U-Net with bilinear')
    else:
        model = WTSNetTimm(model_name=args.name, use_clipping=True).to(device)

    if args.use_sharpness_head:
        try:
            model = SharpnessHead(
                base_model=model,
                in_ch=3, out_ch=3
            ).to(device)
            model.load_model(load_data)
            print('Used full-channels sharpness head')
        except:
            model = SharpnessHeadForYChannel(
                base_model=model,
                in_ch=3, out_ch=3
            ).to(device)
            model.load_model(load_data)
            print('Used Y-channel sharpness head')
    else:
        model.load_state_dict(load_data['model'])

    model.eval()

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    psnr_values = []
    ssim_values = []

    if args.output is None:
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/eval_results_CBSD68/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    clear_folder = os.path.join(args.folder, 'original_png/')

    noisy_folders_and_sigmas: List[Tuple[str, int]] = [
        (os.path.join(args.folder, p), int(p.split('noisy')[1]))
        for p in os.listdir(args.folder)
        if p.startswith('noisy')
    ]

    noisy_folders_and_sigmas.sort(key=lambda s: s[1])

    for noisy_folder, noise_sigma in noisy_folders_and_sigmas:
        if noise_sigma not in [15, 25, 50]:
            continue
    
        print('SIGMA VALUE: {}'.format(noise_sigma))

        output_save_folder = os.path.join(output_folder, 'sigma_{}'.format(noise_sigma))
        os.makedirs(output_save_folder, exist_ok=True)

        dataset_wavelet_losses: List[List[Dict[str, float]]] = []

        for image_name in tqdm(os.listdir(noisy_folder)):
            image_path = os.path.join(noisy_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

            gt_image_path = os.path.join(clear_folder, image_name)
            gt_img = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)

            assert img.shape[0] == gt_img.shape[0] and img.shape[1] == gt_img.shape[1], image_name

            img = pad_image_to_inference(img)
            gt_img = pad_image_to_inference(gt_img)

            input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)

            with torch.no_grad():
                restored_image = eval_denoise_inference(
                    tensor_img=input_tensor, model=model, window_size=imgsz, 
                    batch_size=32, crop_size=imgsz // 32, use_tta=args.use_tta, device=device
                )

            input_tensor = input_tensor.to('cpu')
            pred_image = restored_image.to('cpu')

            if not args.use_sharpness_head:
                dataset_wavelet_losses.append(
                    compute_wavelets_losses(
                        pred_image.unsqueeze(0), 
                        input_tensor.unsqueeze(0), 
                        args.levels
                    )
                )

            pred_image = torch.clamp(pred_image, 0, 1)
            pred_image = tensor_to_image(pred_image)

            rgb_pred = cv2.cvtColor(pred_image, cv2.COLOR_YCrCb2RGB)
            rgb_gt = cv2.cvtColor(gt_img, cv2.COLOR_YCrCb2RGB)

            if args.grayscale:
                ssim_values.append(ssim(cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)))
                psnr_values.append(cv2.PSNR(cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY), cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)))
            else:
                ssim_values.append(ssim(rgb_pred, rgb_gt, channel_axis=2))
                psnr_values.append(cv2.PSNR(rgb_pred, rgb_gt))

            if args.verbose:
                print('Image: {}, PSNR: {:.2f}, SSIM: {:.3f}'.format(image_name, psnr_values[-1], ssim_values[-1]))

            cv2.imwrite(
                os.path.join(output_save_folder, image_name),
                cv2.cvtColor(pred_image, cv2.COLOR_YCrCb2BGR)
            )

        print('Result PSNR -- mean: {:.2f}, std: {:.2f}'.format(np.array(psnr_values).mean(), np.array(psnr_values).std()))
        print('Result SSIM -- mean: {:.3f}, std: {:.3f}'.format(np.array(ssim_values).mean(), np.array(ssim_values).std()))

        if not args.use_sharpness_head:
            print('Values below multiplicated on 1e+4')
            s = 1E4
            for i in range(args.levels):
                print('\nWavelet level #{}: '.format(i + 1))
                _data = {
                    'LH': [],
                    'HL': [],
                    'HH': []
                }

                for waves in dataset_wavelet_losses:
                    for loss_key in _data.keys():
                        _data[loss_key].append(waves[i][loss_key])

                for loss_key in _data.keys():
                    warr = np.array(_data[loss_key], dtype=np.float32)
                    _data[loss_key] = (warr.mean() * s, warr.std() * s)

                    print(
                        '\t{} -- mean: {:.5f}, std: {:.5f}'.format(
                            loss_key, *_data[loss_key]
                        )
                    )
