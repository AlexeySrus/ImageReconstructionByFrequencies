import enum
import torch

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../third_party/traiNNer/codes/'))
from dataops.imresize import resize


class DownSampleMode(enum.Enum):
    MAXPOOL = 1
    BILINEAR = 2
    BICUBIC = 3
    LANCZOS2 = 4
    LANCZOS3 = 5    
    LANCZOS4 = 6
    LANCZOS5 = 5    


class UpSampleMode(enum.Enum):
    CONVTRANSPOSE = 1
    BILINEAR = 2
    BICUBIC = 3
    LANCZOS2 = 4
    LANCZOS3 = 5    
    LANCZOS4 = 6
    LANCZOS5 = 5 


class InterpolationMode(enum.Enum):
    NONE =              (DownSampleMode.MAXPOOL,    UpSampleMode.CONVTRANSPOSE)
    MAXPOOL_BILINEAR =  (DownSampleMode.MAXPOOL,    UpSampleMode.BILINEAR)
    BILINEAR =          (DownSampleMode.BILINEAR,   UpSampleMode.BILINEAR)
    BICUBIC =           (DownSampleMode.BICUBIC,    UpSampleMode.BICUBIC)
    LANCZOS2 =          (DownSampleMode.LANCZOS2,   UpSampleMode.LANCZOS2)
    LANCZOS3 =          (DownSampleMode.LANCZOS3,   UpSampleMode.LANCZOS3)
    LANCZOS4 =          (DownSampleMode.LANCZOS4,   UpSampleMode.LANCZOS4)
    LANCZOS5 =          (DownSampleMode.LANCZOS5,   UpSampleMode.LANCZOS5)


def get_down_function(mode: DownSampleMode) -> torch.nn.Module:
    if mode == DownSampleMode.MAXPOOL:
        return torch.nn.MaxPool2d(2, 2)
    elif mode == DownSampleMode.BILINEAR:
        return lambda x: torch.nn.functional.interpolate(x, scale_factor=0.5, align_corners=False, mode='bilinear')
    elif mode == DownSampleMode.BICUBIC:
        return lambda x: torch.nn.functional.interpolate(x, scale_factor=0.5, align_corners=False, mode='bicubic')
    elif mode == DownSampleMode.LANCZOS2:
        return lambda x: resize(x, scale_factors=0.5, clip=False, interpolation='lanczos2', antialiasing=False)
    elif mode == DownSampleMode.LANCZOS3:
        return lambda x: resize(x, scale_factors=0.5, clip=False, interpolation='lanczos3', antialiasing=False)
    elif mode == DownSampleMode.LANCZOS4:
        return lambda x: resize(x, scale_factors=0.5, clip=False, interpolation='lanczos4', antialiasing=False)
    elif mode == DownSampleMode.LANCZOS5:
        return lambda x: resize(x, scale_factors=0.5, clip=False, interpolation='lanczos5', antialiasing=False)
    else:
        raise RuntimeError('Unsupported mode: {}'.format(mode))


def get_up_function(mode: UpSampleMode, channels: int) -> torch.nn.Module:
    if mode == UpSampleMode.CONVTRANSPOSE:
        return torch.nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
    elif mode == UpSampleMode.BILINEAR:
        return lambda x: torch.nn.functional.interpolate(x, scale_factor=2, align_corners=False, mode='bilinear')
    elif mode == UpSampleMode.BICUBIC:
        return lambda x: torch.nn.functional.interpolate(x, scale_factor=2, align_corners=False, mode='bicubic')
    elif mode == UpSampleMode.LANCZOS2:
        return lambda x: resize(x, scale_factors=2, clip=False, interpolation='lanczos2', antialiasing=False)
    elif mode == UpSampleMode.LANCZOS3:
        return lambda x: resize(x, scale_factors=2, clip=False, interpolation='lanczos3', antialiasing=False)
    elif mode == UpSampleMode.LANCZOS4:
        return lambda x: resize(x, scale_factors=2, clip=False, interpolation='lanczos4', antialiasing=False)
    elif mode == UpSampleMode.LANCZOS5:
        return lambda x: resize(x, scale_factors=2, clip=False, interpolation='lanczos5', antialiasing=False)
    else:
        raise RuntimeError('Unsupported mode: {}'.format(mode))