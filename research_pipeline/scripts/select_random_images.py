from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import os
import numpy as np
from shutil import copyfile

CURRENT_PATH = os.path.dirname(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with dataset which contain images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to result folder with sublsamples frin dataset'
    )
    parser.add_argument(
        '-n', '--number', type=int, required=False, default=1000,
        help='Count of selected files'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files_names = [
        fn
        for fn in os.listdir(args.input)
    ]


    assert len(files_names) > args.number, 'Count of files less then required to save'

    files_names.sort()
    np.random.shuffle(files_names)

    for file_to_delete in tqdm(files_names[:args.number]):
        image_path = os.path.join(args.input, file_to_delete)
        copy_path = os.path.join(args.output, file_to_delete)

        copyfile(image_path, copy_path)

        os.remove(image_path)
