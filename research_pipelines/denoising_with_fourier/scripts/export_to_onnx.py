from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import torch
import os
import scipy.io as sio
from timeit import default_timer as time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CURRENT_PATH = os.path.dirname(__file__)

from FFTCNN.combined_attn_unet import FFTAttentionUNet as FFTCNN
from FFTCNN.uformer import Uformer
from FFTCNN.interpolation_type import interpolation_type_from_str



def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--attention_mode', type=str, required=False, default='full',
        choices=['full', 'ca', 'sa', 'fca', 'cbam', 'none'],
        help='Attention mode from \'full\', \'ca\', \'sa\', \'fca\', \'cbam\', \'none\''
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to output ONNX model'
    )
    parser.add_argument(
        '-j', '--jit', type=str, required=False,
        help='Path to JIT traced model (Optional)'
    )
    parser.add_argument(
        '--interpolation_mode', type=str, required=False, default='none',
        choices=['none', 'max2bilinear', 'bilinear', 'bicubic', 'lanczos2', 'lanczos3', 'lanczos4', 'lanczos5'],
        help='Interpolation mode where \'none\' is classic maxpool-decomvolution scheme, '
                '\'max2bilinear\' is maxpool-bilinear scheme and other is X-X downsample-upsample interpolations methods.'
    )
    return parser.parse_args()


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img



if __name__ == '__main__':
    args = parse_args()

    in_ch = 3
    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FFTCNN(
        use_substraction=True, 
        attention_mode=args.attention_mode,
        interolation_mode=interpolation_type_from_str(args.interpolation_mode)
    ).to(device)
    # model.forward = model.custom_forward

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)
    model.load_state_dict(load_data['model'])
    model.eval()
    model.to_export()

    if args.jit is not None:
        traced_model = torch.jit.trace(
            model,
            torch.rand(1, in_ch, imgsz, imgsz, device=device, requires_grad=True)
        )
        traced_model.save(args.jit)

    # load_data = torch.load(args.model, map_location=torch.device(device))
    # model = Uformer(
    #     img_size=imgsz, embed_dim=32, win_size=8, 
    #     token_projection='linear', token_mlp='leff',
    #     depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True,
    #     dd_in=1, in_chans=1,
    #     attention_mode=args.attention_mode
    # ).to(device)

    # model.load_state_dict(load_data['model'])
    # model.eval()

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    torch.onnx.export(
        model,
        torch.rand(1, in_ch, imgsz, imgsz, device=device, requires_grad=True),
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
        do_constant_folding=True,
        opset_version=16
    )
