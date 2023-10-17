from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CURRENT_PATH = os.path.dirname(__file__)

from IS_Net.wt_attn_isnet import ISNetWT


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '-i', '--image', type=str, required=False,
        help='Path to input image (optional)'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with output visualizations (optional)'
    )
    parser.add_argument(
        '-g', '--gamma_correction', type=float, required=False, default=1.5,
        help='Coefficient to adjust gamma of wavelets coefficients visualization (optional)'
    )
    return parser.parse_args()


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


def add_paddings_to_image(image: np.ndarray, border: int = 8) -> np.ndarray:
    return np.pad(
        image, 
        ((border, border), (border, border), (0, 0)), 
        mode='constant', constant_values=255
    )


def build_wavelets_visualization(wavelets_parts: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], size: int = 1024) -> np.ndarray:
    ll, lh, hl, hh = wavelets_parts
    res = np.concatenate(
        (
            np.concatenate((ll, lh), axis=1),
            np.concatenate((hl, hh), axis=1)
        ),
        axis=0
    )
    res = cv2.resize(res, (size, size), interpolation=cv2.INTER_CUBIC)
    return res


if __name__ == '__main__':
    args = parse_args()

    model = ISNetWT(in_ch=3, out_ch=3 * 4, image_ch=3)

    load_path = args.model
    load_data = torch.load(load_path, map_location='cpu')
    model.load_state_dict(load_data['model'])

    print('Best PSNR: {:.2f}'.format(load_data['acc']))

    if args.output is None:
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/results/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    if args.image is None:
        image_path = os.path.join(CURRENT_PATH, '../../../materials/sony_a7c_5_crop.png')
    else:
        image_path = args.image

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
    
    d1, d2, d3, d4, d5, d6 = outputs[0]

    final_vis_grid = []

    for wavelets_level, w in enumerate([d2, d3, d4, d5, d6]):
        ll, lh, hl, hh = torch.split(w[0], 3, dim=0)
        ll = torch.clamp(ll / 2, 0, 1)
        lh = torch.clamp(lh + 0.5, 0, 1)
        hl = torch.clamp(hl + 0.5, 0, 1)
        hh = torch.clamp(hh + 0.5, 0, 1)

        ll = tensor_to_image(ll)
        lh = tensor_to_image(lh)
        hl = tensor_to_image(hl)
        hh = tensor_to_image(hh)

        lh = adjust_gamma(lh, args.gamma_correction)
        hl = adjust_gamma(lh, args.gamma_correction)
        hh = adjust_gamma(lh, args.gamma_correction)

        ll = add_paddings_to_image(ll)
        lh = add_paddings_to_image(lh)
        hl = add_paddings_to_image(hl)
        hh = add_paddings_to_image(hh)

        cv2.imwrite(
            os.path.join(output_folder, 'LL_{}.png'.format(wavelets_level + 1)),
            ll
        )
        cv2.imwrite(
            os.path.join(output_folder, 'LH_{}.png'.format(wavelets_level + 1)),
            lh
        )
        cv2.imwrite(
            os.path.join(output_folder, 'HL_{}.png'.format(wavelets_level + 1)),
            hl
        )
        cv2.imwrite(
            os.path.join(output_folder, 'HH_{}.png'.format(wavelets_level + 1)),
            hh
        )

        vis_img = build_wavelets_visualization((ll, lh, hl, hh))
        cv2.imwrite(
            os.path.join(output_folder, 'full_{}.png'.format(wavelets_level + 1)),
            vis_img
        )

        final_vis_grid.append(vis_img)


    final_vis_grid = np.concatenate(final_vis_grid, axis=1)
    cv2.imwrite(
        os.path.join(output_folder, 'final_grid.png'),
        final_vis_grid
    )