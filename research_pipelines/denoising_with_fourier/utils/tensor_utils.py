import numpy as np
import torch
from enum import Enum
import cv2


def preprocess_image(img: np.ndarray, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    return (torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0 - mean) / std   # by default from 0..255 to -1..1


class TensorRotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    HORISONTAL_FLIP = lambda x: x.flip(2)
    ROTATE_90_CLOCKWISE = lambda x: x.transpose(1, 2).flip(2)
    ROTATE_180 = lambda x: x.flip(1, 2)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: x.transpose(1, 2).flip(1)


def rotate_tensor(img: torch.Tensor, rot_value: TensorRotate) -> torch.Tensor:
    """Rotate image tensor

    Args:
        img: tensor in CHW format
        rot_value: element of TensorRotate class, possible values
            TensorRotate.NONE,
            TensorRotate.HORISONTAL_FLIP,
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_180,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE,

    Returns:
        Rotated image in same of input format
    """
    return rot_value(img)


def from_torch_fft_to_np(x: torch.Tensor) -> np.ndarray:
    fshift = x.permute(0, 2, 3, 1).numpy()
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    magnitude_spectrum = np.pad(
        magnitude_spectrum,
        ((0, 0), (0, 0), (x.size(3), 0), (0, 0)),
        mode='reflect'
    )
    magnitude_spectrum[:, :x.size(2) // 2] = magnitude_spectrum[:, :x.size(2) // 2][:, ::-1]
    magnitude_spectrum[:, x.size(2) // 2:] = magnitude_spectrum[:, x.size(2) // 2:][:, ::-1]
    return magnitude_spectrum.transpose(0, 3, 1, 2)
