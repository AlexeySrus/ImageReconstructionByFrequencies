from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import os
import numpy as np

CURRENT_PATH = os.path.dirname(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with dataset which contain \'clear\' and \'noisy\' subfolders'
    )
    parser.add_argument(
        '-n', '--number', type=str, required=False, default=1000,
        help='Count of saving files'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    clear_folder = os.path.join(args.input, 'clear/')
    noisy_folder = os.path.join(args.input, 'noisy/')

    clear_files_names = [
        fn
        for fn in os.listdir(clear_folder)
    ]

    noisy_warp = {
        os.path.splitext(fn)[1]: os.path.join(noisy_folder, fn)
        for fn in os.listdir(noisy_folder)
    }

    assert len(clear_files_names > args.number), 'Count of files less then required to save'

    clear_files_names.sort()
    np.random.shuffle(clear_files_names)

    for file_to_delete in tqdm(clear_files_names[args.number:]):
        clear_image_path = os.path.join(clear_folder, file_to_delete)
        noisy_image_path = noisy_warp[os.path.splitext(file_to_delete)[0]]

        os.remove(clear_image_path)
        os.remove(noisy_image_path)
