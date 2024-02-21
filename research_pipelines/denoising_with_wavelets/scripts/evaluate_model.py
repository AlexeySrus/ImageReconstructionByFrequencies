from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch
import os

CURRENT_PATH = os.path.dirname(__file__)

from WTSNet.wts_timm import UnetTimm, WTSNetTimm, SharpnessHead
from utils.window_inference import eval_denoise_inference, denoise_inference
from utils.cas import contrast_adaptive_sharpening


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
    return parser.parse_args()
    

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


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
        model = SharpnessHead(
            base_model=model,
            in_ch=3, out_ch=3
        ).to(device)
        model.load_model(load_data)
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
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = tensor_to_image(pred_image)

        # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2YCrCb)

        ssim_values.append(ssim(pred_image[..., 0], gt_img[..., 0]))
        psnr_values.append(cv2.PSNR(pred_image[..., 0], gt_img[..., 0]))

        if args.verbose:
            print('Image: {}, PSNR: {:.2f}, SSIM: {:.3f}'.format(image_name, psnr_values[-1], ssim_values[-1]))

        cv2.imwrite(
            os.path.join(output_folder, image_name),
            cv2.cvtColor(pred_image, cv2.COLOR_YCrCb2BGR)
        )

    avg_psnr = np.array(psnr_values).mean()
    avg_ssim = np.array(ssim_values).mean()
    print('Result PSNR: {:.2f}'.format(avg_psnr))
    print('Result SSIM: {:.3f}'.format(avg_ssim))
