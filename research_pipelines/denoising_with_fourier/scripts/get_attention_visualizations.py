from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import h5py
from tqdm import tqdm
import torch
import os
import scipy.io as sio
from timeit import default_timer as time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
        '-i', '--image', type=str, required=True,
        help='Path to image'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to folder with output visualizations (optional)'
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
    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    model = FFTCNN(use_substraction=True).to(device)

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)
    model.load_state_dict(load_data['model'])
    model.eval()

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    image_path = args.image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[0] != imgsz or img.shape[1] != imgsz:
        print('Resized from {}'.format(img.shape[:2]))
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

    input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0).unsqueeze(0)
    input_tensor.requires_grad = False

    with torch.no_grad():
        pred, attn_maps = model(input_tensor.to(device))

    pred = torch.clamp(pred, 0, 1)
    pred_image = tensor_to_image(pred[0].to('cpu'))
    attn_maps = torch.cat(attn_maps, dim=1).to('cpu')
    attn_maps = tensor_to_image(attn_maps[0])

    cv2.imwrite(
        os.path.join(output_folder, 'output.png'),
        cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
    )

    maps_names = [
        'fourier_map_1',
        'wt_maps_1',
        'fourier_map_2',
        'wt_maps_2',
        'fourier_map_3',
        'wt_maps_3',
        'fourier_map_4',
        'wt_maps_4'
    ]

    for i in range(len(maps_names)):
        save_path = os.path.join(output_folder, '{}.png'.format(maps_names[i]))
        cv2.imwrite(
            save_path,
            attn_maps[..., i]
        )

    # model.to_export()
    # torch.onnx.export(
    #     model,
    #     torch.rand(1, 3, imgsz, imgsz, device=device, requires_grad=True),
    #     os.path.join(output_folder, 'sfwaunet.onnx'),
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         "input": {0: "batch"},
    #         "output": {0: "batch"},
    #     },
    #     opset_version=16
    # )
