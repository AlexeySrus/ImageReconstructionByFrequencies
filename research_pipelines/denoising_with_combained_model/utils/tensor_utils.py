import numpy as np
import torch
from enum import Enum


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



### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).to(rgb_gt.device)

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy
