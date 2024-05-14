"""Good differentiable image resampling for PyTorch."""

from typing import Optional, Tuple, Union

from functools import update_wrapper
import math

import torch
from torch.nn import functional as F
import numpy as np


def sinc(x):
    return torch.where(x != 0,  torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a <= x, x < a)
    out = torch.where(cond, a * sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def odd(fn):
    return update_wrapper(lambda x: torch.sign(x) * fn(abs(x)), fn)


def _to_linear_srgb(input):
    cond = input <= 0.04045
    a = input / 12.92
    b = ((input + 0.055) / 1.055)**2.4
    return torch.where(cond, a, b)


def _to_nonlinear_srgb(input):
    cond = input <= 0.0031308
    a = 12.92 * input
    b = 1.055 * input**(1/2.4) - 0.055
    return torch.where(cond, a, b)


to_linear_srgb = odd(_to_linear_srgb)
to_nonlinear_srgb = odd(_to_nonlinear_srgb)


def downsample_lanczos(input: torch.Tensor, size: Optional[Tuple[int, int]] = None, 
                     scale: Optional[Union[Tuple[float, float], Tuple[int, int], float, int]] = None, 
                     align_corners: bool = True, is_srgb: bool = False,
                     a: float = 3) -> torch.Tensor:
    if size is None and scale is None:
        raise RuntimeError('Need to set size or scale')
    if a < 0:
        raise RuntimeError('Invalid a = {}, a must be > 0'.format(a))
    
    n, c, h, w = input.shape

    if size is not None:
        dh, dw = size
    elif isinstance(scale, float) or isinstance(scale, int):
        dh = int(h * scale)
        dw = int(w * scale)
    else:
        dh = int(h * scale[0])
        dw = int(w * scale[1])

    if is_srgb:
        input = to_linear_srgb(input)

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, a), a).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, a), a).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    input = F.interpolate(input, size, scale_factor=scale, mode='bicubic', align_corners=align_corners)

    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input


def get_kernel_range(a: int) -> torch.Tensor:
    return torch.arange(-a + 1, a + 1, 1)

def get_lanczos_kernel(k: int, a: int) -> torch.Tensor:
    k_range = 1.0 / float(k) - get_kernel_range(a).to(torch.float32)
    lanczos_k = lanczos(k_range, a)
    return lanczos_k


def resample_lanczos(input: torch.Tensor, size: Optional[Tuple[int, int]] = None, 
                     scale: Optional[Union[Tuple[float, float], Tuple[int, int], float, int]] = None, 
                     align_corners: bool = True, is_srgb: bool = False,
                     a: float = 3) -> torch.Tensor:
    if size is None and scale is None:
        raise RuntimeError('Need to set size or scale')
    if a < 0:
        raise RuntimeError('Invalid a = {}, a must be > 0'.format(a))
    
    n, c, h, w = input.shape

    if size is not None:
        dh, dw = size
    elif isinstance(scale, float) or isinstance(scale, int):
        dh = int(h * scale)
        dw = int(w * scale)
    else:
        dh = int(h * scale[0])
        dw = int(w * scale[1])

    assert dh == dw, 'Only supported square interpolations'

    scale = dh / input.size(2)
    if scale < 1:
        return downsample_lanczos(input, size=(dh, dw), align_corners=align_corners, is_srgb=is_srgb, a=a)
    
    scale = int(scale)

    if is_srgb:
        input = to_linear_srgb(input)

    input = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=align_corners)

    input = input.view([n * c, 1, dh, dw])

    kernel = get_lanczos_kernel(k, a).numpy()
    kernel_2d = torch.from_numpy(np.outer(kernel, kernel.T)).to(input.device)

    pad_v = kernel_2d.size(0) // 2
    input = F.pad(input, (pad_v, pad_v, pad_v, pad_v), 'reflect')
    input = F.conv2d(input, kernel_2d[None, None])

    input = input.view([n, c, dh, dw])
    
    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input


if __name__ == '__main__':

    a = 3
    k = 2
    N = 64
    c = 3
    sign = torch.rand(1, c, N, N)
    n = sign.size(2)



    isign = torch.nn.functional.interpolate(sign, scale_factor=k, mode='bilinear', align_corners=True)

    print(torch.linalg.norm(isign - r).max())
