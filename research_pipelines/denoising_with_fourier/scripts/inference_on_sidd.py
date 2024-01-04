import numpy as np
import os,sys
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from einops import rearrange, repeat

import cv2
import torch

import scipy.io as sio
import utils
import math

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from FFTCNN.combined_attn_unet import FFTAttentionUNet as FFTCNN
from utils.window_inference import eval_denoise_inference


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CURRENT_PATH = os.path.dirname(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '-f', '--folder', type=str, required=True,
        help='Path to folder with the Darmstadt Noise Dataset'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with output visualizations (optional)'
    )
    return parser.parse_args()


def RGB2YCrCb(t: torch.Tensor) -> torch.Tensor:
    _img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2YCrCb)
    res_t = torch.from_numpy(_img.astype(np.float32) / 255.0)
    return res_t.permute(2, 0, 1)


def YCrCb2RGB(t: torch.Tensor) -> torch.Tensor:
    _img = (torch.clamp(t, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    _img = cv2.cvtColor(_img, cv2.COLOR_YCrCb2RGB)
    res_t = torch.from_numpy(_img.astype(np.float32) / 255.0)
    return res_t.permute(2, 0, 1)


def YCrCb2NPRGB(t: torch.Tensor) -> np.ndarray:
    _img = (torch.clamp(t, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    _img = cv2.cvtColor(_img, cv2.COLOR_YCrCb2RGB)
    return _img


if __name__ == '__main__':
    args = parse_args()

    if args.output is None:
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/sidd_fftcnn_inference_results/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    frames_folder = os.path.join(output_folder, 'pred/')
    os.makedirs(frames_folder, exist_ok=True)

    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Device for inference: {}'.format(device))

    model = FFTCNN().to(device)

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)
    model.load_state_dict(load_data['model'])
    model.eval()
    # model.to_export()

    # Fake run to optimize
    with torch.no_grad():
        _ = model(torch.rand(1, 3, imgsz, imgsz).to(device))

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    boxes_data = sio.loadmat(os.path.join(args.folder, 'BenchmarkBlocks32.mat'))['BenchmarkBlocks32']
    n_boxes = len(boxes_data)

    dataset_folders = [p for p in os.listdir(args.folder) if os.path.isdir(os.path.join(args.folder, p))]
    dataset_folders.sort(key=lambda fname: int(fname.split('_')[0]))

    restored = np.zeros(shape=(len(dataset_folders), n_boxes, 256, 256, 3), dtype=np.uint8)

    with torch.no_grad():
        for i, folder_name in enumerate(tqdm(dataset_folders)):
            base_id_str = folder_name.split('_')[0]
            image_path = os.path.join(args.folder, folder_name, '{}_NOISY_SRGB_010.PNG'.format(base_id_str))
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

            torch_image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)

            for k in range(n_boxes):
                y, x, h, w = boxes_data[k]

                noisy_patch = torch_image[:, y:y+h, x:x+w]

                restored_patch = eval_denoise_inference(
                    tensor_img=noisy_patch, model=model, window_size=imgsz, 
                    batch_size=4, crop_size=0, use_tta=True, device=device
                ).to('cpu')

                restored_patch = torch.clamp(restored_patch, 0, 1)
                restored_patch = (restored_patch.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

                output_path = os.path.join(
                    frames_folder,
                    '{}_{}_{}.png'.format(folder_name, i, k)
                )
                cv2.imwrite(
                    output_path,
                    cv2.cvtColor(restored_patch, cv2.COLOR_YCrCb2BGR)
                )

                restored[i, k, :,:,:] = cv2.cvtColor(restored_patch, cv2.COLOR_YCrCb2RGB)

            torch_image = torch_image.to('cpu')
            del torch_image

    # save denoised data
    sio.savemat(os.path.join(output_folder, 'SubmitSrgb.mat'), {"Idenoised": restored,})
