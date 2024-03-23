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
        help='Path to folder with files'
    )
    parser.add_argument(
        '-n', '--number', type=str, required=False, default=1000,
        help='Count of saving files'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    files_names = [
        os.path.join(args.input, fn)
        for fn in os.listdir(args.input)
    ]

    assert len(files_names > args.number), 'Count of files less then required to save'

    files_names.sort()
    np.random.shuffle(files_names)

    for file_to_delete in tqdm(files_names[args.number:]):
        os.remove(file_to_delete)
