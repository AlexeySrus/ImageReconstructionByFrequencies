from typing import Tuple, List, Dict, Callable, Optional
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
from dipy.denoise.noise_estimate import piesno

CURRENT_PATH = os.path.dirname(__file__)
NOISE_SIGMA_THRESHOLD: float = 7.5


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with .h5 files'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to folder with output images'
    )
    parser.add_argument(
        '--crop', action='store_true',
        help='Apply crop of useful area'
    )
    parser.add_argument(
        '--minsize', type=int, required=False, default=256,
        help='Minimum size of frame to save'
    )
    parser.add_argument(
        '--select-noisy', action='store_true',
        help='Save only images with noise'
    )
    # parser.add_argument(
    #     '-t', '--type', type=str, required=False, default='singlecoil',
    #     choices=['singlecoil', 'multicoil'],
    #     help='Type of MRI image: \'singlecoil\' or \'multicoil\''
    # )
    return parser.parse_args()


def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(np.abs(sd) < 1E-5, 0, m/sd)


def crop_by_binary_mask(channel: np.ndarray, threshold: int = 15) -> Optional[np.ndarray]:
    _, bimg = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
    x1, y1, w, h = cv2.boundingRect(bimg)
    x2 = x1 + w
    y2 = y1 + h

    d = 15
    x1 = max(0, x1 - d)
    y1 = max(0, y1 - d)
    x2 = min(bimg.shape[1] - 1, x2 + d)
    x2 = min(bimg.shape[0] - 1, y2 + d)

    if (x2 - x1) * (y2 - y1) == 0:
        return None

    img_to_save = channel[y1:y2, x1:x2].copy()
    return img_to_save


if  __name__ == '__main__':
    args = parse_args()

    use_crop = args.crop

    base_folder_name = os.path.basename(str(args.input).rstrip('/'))

    os.makedirs(args.output, exist_ok=True)

    for fname in tqdm(os.listdir(args.input)):
        bname, ext = os.path.splitext(fname)
        if ext != '.h5':
            continue

        fpath = os.path.join(args.input, fname)

        try:
            hf = h5py.File(fpath)
        except Exception as e:
            print('Scipt proecss file {} because: {}'.format(fname, e))
            continue

        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        masked_kspace = transforms.to_tensor(hf["kspace"][()])

        multicoil_reconstruction_rss = hf["reconstruction_rss"][:]

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

        sigma_arr = piesno(nimg, N=4, return_mask=False)

        if isinstance(sigma_arr, np.ndarray):
            sigma = sigma_arr.max()
        else:
            sigma = sigma_arr

        if args.select_noisy:
            if sigma < NOISE_SIGMA_THRESHOLD:
                continue
        else:
            if sigma > 0.9:
                continue

        for slice_id in range(nimg.shape[0]):
            res_path = os.path.join(
                args.output, 
                '{}_{}_slice_{}.png'.format(base_folder_name, bname, slice_id)
            )

            img_to_save = None

            if len(nimg.shape) == 3:
                # singlecoil type 
                img_to_save = nimg[slice_id].copy()
            elif len(nimg.shape) == 4:
                # multicoil type 
                sigma = sigma_arr[slice_id]
                if args.select_noisy:
                    if sigma < NOISE_SIGMA_THRESHOLD:
                        continue
                else:
                    if sigma > 0.9:
                        continue

                img_to_save = multicoil_reconstruction_rss[slice_id]

                img_to_save = (img_to_save - img_to_save.min()) / (img_to_save.max() - img_to_save.min())
                img_to_save = (img_to_save * 255.0).astype(np.uint8)
            else:
                raise RuntimeError('Not supported shape: {}'.format(nimg.shape))
            
            snr = signaltonoise(img_to_save)
            if np.abs(1.0 - snr) > 0.3:
                continue

            if use_crop:
                img_to_save = crop_by_binary_mask(img_to_save)
                if img_to_save is None:
                    continue

            if min(img_to_save.shape[:2]) < args.minsize:
                continue

            is_save = cv2.imwrite(
                res_path,
                img_to_save,
                [cv2.IMWRITE_PNG_COMPRESSION, 0]
            )
            assert is_save, res_path

        hf.close()
