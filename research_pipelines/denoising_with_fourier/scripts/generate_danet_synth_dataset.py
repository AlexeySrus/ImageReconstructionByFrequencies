from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import torch
from tqdm import tqdm
from shutil import copyfile
import os
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
        '-i', '--input', type=str, required=True,
        help='Path to folder with images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to folder with training/val set'
    )
    parser.add_argument(
        '--noise_generator_weights', type=str, required=True,
        help='Path to file of DANet Generator weights'
    )
    parser.add_argument(
        '--minsize', type=int, required=False, default=256,
        help='Minimum image size (default: 256)'
    )
    parser.add_argument(
        '--prefix', type=str, required=False,
        help='Prefix of noise images names (default: None)'
    )
    return parser.parse_args()


def danet_inference(inp_tensor: torch.Tensor, net_g: torch.nn.Module) -> torch.Tensor:
    out = sample_generator(net_g, inp_tensor)
    out = torch.clamp(out, 0.0, 1.0)
    return out


if  __name__ == '__main__':
    args = parse_args()

    output_clear_images_folder = os.path.join(args.output, 'clear/')
    output_noisy_images_folder = os.path.join(args.output, 'noisy/')

    os.makedirs(output_clear_images_folder, exist_ok=True)
    os.makedirs(output_noisy_images_folder, exist_ok=True)

    interpolations = [
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4,
        cv2.INTER_LINEAR
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    net = UNetG(3, wf=32, depth=5).to(device)
    # net = get_base_uformer_model(image_size=256, in_ch=4, out_ch=3).to(device)
    net.load_state_dict(torch.load(args.noise_generator_weights, map_location=device)['G'])

    for img_name in tqdm(os.listdir(args.input)):
        bname, ext = os.path.splitext(img_name)

        input_path = os.path.join(args.input, img_name)

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)

        if min(img.shape[:2]) < args.minsize:
            k = args.minsize / min(img.shape[:2])
            img = cv2.resize(
                img, None, fx=k, fy=k, 
                interpolation=np.random.choice(interpolations)
            )

        timg = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0).to(device)
        inference_func = lambda tx: danet_inference(tx, net)
        with torch.no_grad():
            nimg = denoise_inference(
                timg,
                inference_func,
                256,
                16,
                crop_size=16
            )
        nimg = (nimg.to('cpu') * 255.0).numpy().transpose(1, 2, 0).astype(np.uint8)

        if args.prefix is not None:
            image_basename = '{}_{}'.format(
                args.prefix, bname
            )
        else:
            image_basename = bname
        
        clear_path = os.path.join(
            output_clear_images_folder,
            image_basename + ext
        )
        noisy_path = os.path.join(
            output_noisy_images_folder,
            image_basename + ext
        )

        copyfile(input_path, clear_path)
        is_save = cv2.imwrite(
            noisy_path,
            nimg,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
        assert is_save, noisy_path
