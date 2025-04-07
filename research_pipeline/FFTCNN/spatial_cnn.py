from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn


padding_mode: str = 'reflect'


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def convert_weights_from_old_version(_weights: OrderedDict) -> OrderedDict:
    new_weights = OrderedDict(
        [('wtsmodel.' + k, v) if not k.startswith('wtsmodel.') else (k, v) for k, v in _weights.items()]
    )
    return new_weights


class DownscaleByWaveletes(nn.Module):
    def __init__(self) -> None:
        """
        Downscale by Haar Wavelets transforms
        This transform increase channels by 4 times: C -> C * 4
        """
        super().__init__()

        self.dwt = HaarForward()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.dwt(x)
        step = out.size(1) // 4
        return out[:, :step], out[:, step:]


class UpscaleByWaveletes(nn.Module):
    def __init__(self) -> None:
        """
        Downscale by Haar Wavelets transforms
        This transform reduce channels by 4 times: C -> C // 4
        """
        super().__init__()

        self.iwt = HaarInverse()

    def forward(self, x_ll: torch.Tensor, hight_freq: torch.Tensor) -> torch.Tensor:
        out = self.iwt(torch.cat((x_ll, hight_freq), dim=1))
        return out
    

def conv1x1(in_ch, out_ch):
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=1,
        stride=1,
        padding=0
    )


def conv3x3(in_ch, out_ch):
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode=padding_mode
    )


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.act1 = nn.ReLU()
        self.conv2 = conv3x3(in_ch * 2, out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y
    

class FeaturesProcessingWithLastConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch)
        self.final_conv = conv1x1(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.final_conv(y)
        return y


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.pool(y)
        return y
    

class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.act = nn.Mish()
        self.features = FeaturesProcessing(in_ch // 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y = self.act(y)
        y = self.features(y)
        return y


class SpatialMiniUNet(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch)

        self.final_conv = conv1x1(out_ch, out_ch)

        self.conv_out = FeaturesProcessingWithLastConv(out_ch, out_ch)
        self.up_conv_out = FeaturesProcessingWithLastConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.init_block(x)

        down_f1 = self.downsample_block1(hx)
        down_f2 = self.downsample_block2(down_f1)

        deep_f = self.deep_conv_block(down_f2)

        deep_f = self.upsample2(deep_f)
        decoded_f2 = torch.cat((down_f1, deep_f), axis=1)
        decoded_f2 = self.upsample_features_block2(decoded_f2)

        decoded_f2 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, decoded_f2), dim=1) 
        decoded_f1 = self.upsample_features_block1(decoded_f1)

        decoded_f1 = self.final_conv(decoded_f1)

        return decoded_f1



if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.onnx import OperatorExportTypes
    from fvcore.nn import FlopCountAnalysis
    # from pthflops import count_ops
    from pytorch_optimizer import AdaSmooth