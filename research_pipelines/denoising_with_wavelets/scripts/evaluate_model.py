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
        '--use_unet', action='store_true',
        help='Use classic U-Net network decoder architecture'
    )
    parser.add_argument(
        '--use_tta', action='store_true',
        help='Use test time augmentations until inference'
    )
    parser.add_argument(
        '--use_sharpness_head', action='store_true',
        help='Use sharpness head'
    )
    parser.add_argument(
        '-l', '--levels', type=int, required=False, default=5,
        help='Wavelte levels'
    )
    return parser.parse_args()
    

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
    device = 'cuda'

    if args.use_unet:
        model = UnetTimm(model_name=args.name).to(device)
    else:
        model = WTSNetTimm(model_name=args.name, use_clipping=True).to(device)

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)

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
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/eval_results/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    noisy_folder = os.path.join(args.folder, 'noisy/')
    clear_folder = os.path.join(args.folder, 'clear/')

    dataset_wavelet_losses: List[List[Dict[str, float]]] = []

    for image_name in tqdm(os.listdir(noisy_folder)):
        image_path = os.path.join(noisy_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        gt_image_path = os.path.join(clear_folder, image_name)
        gt_img = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)

        input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)

        with torch.no_grad():
            restored_image = eval_denoise_inference(
                tensor_img=input_tensor, model=model, window_size=imgsz, 
                batch_size=32, crop_size=imgsz // 32, use_tta=args.use_tta, device=device
            )

        input_tensor = input_tensor.to('cpu')

        pred_image = restored_image.to('cpu')

        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2YCrCb)

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

        ssim_values.append(ssim(pred_image[..., 0], gt_img[..., 0]))
        psnr_values.append(cv2.PSNR(pred_image[..., 0], gt_img[..., 0]))

        if args.verbose:
            print('Image: {}, PSNR: {:.2f}, SSIM: {:.3f}'.format(image_name, psnr_values[-1], ssim_values[-1]))

        cv2.imwrite(
            os.path.join(output_folder, image_name),
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
