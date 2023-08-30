import numpy as np
import torch


def preprocess_image(img: np.ndarray, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    return (torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0 - mean) / std   # by default from 0..255 to -1..1
