from typing import Tuple, List

import torch
import torch.nn as nn

from WTSNet.cbam import CBAM
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


padding_mode: str = 'reflect'


class DownscaleByWaveletes(nn.Module):
    def __init__(self, wavename: str = 'haar') -> None:
        """
        Downscale by Wavelets transforms
        This transform increase channels by 4 times: C -> C * 4

        Args:
            wavename (str, optional): wavelet name. Defaults to 'haar'.
        """
        super().__init__()

        self.dwt = DWT_2D(wavename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll, x_lh, x_hl, x_hh = self.dwt(x)
        out = torch.cat((x_ll, x_lh, x_hl, x_hh), dim=1)
        return out


class UpscaleByWaveletes(nn.Module):
    def __init__(self, wavename: str = 'haar') -> None:
        """
        Downscale by Wavelets transforms
        This transform reduce channels by 4 times: C -> C // 4

        Args:
            wavename (str, optional): wavelet name. Defaults to 'haar'.
        """
        super().__init__()

        self.iwt = IDWT_2D(wavename)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll, x_lh, x_hl, x_hh = torch.split(x, x.size(1) // 4, dim=1)
        out = self.iwt(x_ll, x_lh, x_hl, x_hh)
        return out


class ConvModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ksize: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            padding_mode=padding_mode
        )
        self.act = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.act
        return y
    


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvModule(in_ch, in_ch * 2, 5)



class WTSNet(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()

        out_w_ch = image_channels * 4

        self.in_dwt = DWT_2D()

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        in_x_ll, in_x_lh, in_x_hl, in_x_hh = self.in_dwt(x)



