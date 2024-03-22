from typing import Tuple, List, Dict, Callable
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
from tqdm import tqdm
import torch
import os

CURRENT_PATH = os.path.dirname(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to folder with images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with training/val set'
    )
    parser.add_argument(
        '--grayscale', action='store_true',
        help='Use grayscale images to evaluate'
    )
    return parser.parse_args()


if  __name__ == '__main__':
    args = parse_args()

    noisy_sigmas = [15, 25, 50, 75, 150]