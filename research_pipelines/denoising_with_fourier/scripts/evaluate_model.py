from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import torch
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CURRENT_PATH = os.path.dirname(__file__)

from FFTCNN.combined_attn_unet import FFTAttentionUNet as FFTCNN
# from FFTCNN.unet import AttentionUNet as FFTCNN
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
        '--y-channel-only', action='store_true',
        help='Evaluate only on Y component from YCrCb color space'
    )
    return parser.parse_args()
    

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


if __name__ == '__main__':
    args = parse_args()

    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_only = args.y_channel_only

    model = FFTCNN().to(device)

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)
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

        input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)
        
        with torch.no_grad():
            restored_image = eval_denoise_inference(
                tensor_img=input_tensor, model=model, window_size=imgsz, 
                batch_size=4, crop_size=imgsz // 32, use_tta=True, device=device
            )

        input_tensor = input_tensor.to('cpu')

        pred_image = restored_image.to('cpu')
        pred_image = torch.clamp(pred_image, 0, 1)
        pred_image = tensor_to_image(pred_image)

        if y_only:
            psnr_values.append(
                cv2.PSNR(
                    pred_image[..., 0], 
                    cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[..., 0]
                )
            )
        else:
            psnr_values.append(
                cv2.PSNR(
                    cv2.cvtColor(pred_image, cv2.COLOR_YCrCb2BGR), 
                    gt_img
                )
            )
            
        ssim_values.append(
            ssim(
                pred_image[..., 0], 
                cv2.cvtColor(gt_img, cv2.COLOR_BGR2YCrCb)[..., 0]
            )
        )

        print('Image: {}, PSNR: {:.2f}'.format(image_name, psnr_values[-1]))

        cv2.imwrite(
            os.path.join(output_folder, image_name),
            cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        )

    avg_psnr = np.array(psnr_values).mean()
    avg_ssim = np.array(ssim_values).mean()
    print('Result PSNR: {:.2f}'.format(avg_psnr))
    print('Result SSIM: {:.3f}'.format(avg_ssim))
