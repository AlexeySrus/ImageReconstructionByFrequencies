from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import fastmri
from fastmri.data import transforms
from fastmri.data.mri_data import et_query
from tqdm import tqdm
import os
import h5py
import xml.etree.ElementTree as etree

CURRENT_PATH = os.path.dirname(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with .h5 files'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with output images'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    base_folder_name = os.path.basename(str(args.input).rstrip('/'))

    os.makedirs(args.output, exist_ok=True)

    for fname in tqdm(os.listdir(args.input)):
        bname, ext = os.path.splitext(fname)
        if ext != '.h5':
            continue

        fpath = os.path.join(args.input, fname)

        with h5py.File(fpath, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            masked_kspace = transforms.to_tensor(hf["kspace"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            crop_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
            )

            image = fastmri.ifft2c(masked_kspace)

            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            image = transforms.complex_center_crop(image, crop_size)
            image = fastmri.complex_abs(image)

            nimg = (image - image.min()) / (image.max() - image.min())
            nimg = (nimg * 255.0).numpy().astype(np.uint8)

            res_path = os.path.join(
                args.output, 
                '{}_{}.png'.format(base_folder_name, bname)
            )

            is_save = cv2.imwrite(
                res_path,
                nimg,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            assert is_save, res_path
