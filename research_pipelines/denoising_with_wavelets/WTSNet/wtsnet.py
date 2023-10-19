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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_ll, x_lh, x_hl, x_hh = self.dwt(x)
        hight_freq = torch.cat((x_lh, x_hl, x_hh), dim=1)
        return x_ll, hight_freq


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

    def forward(self, x_ll: torch.Tensor, hight_freq: torch.Tensor) -> torch.Tensor:
        x_lh, x_hl, x_hh = torch.split(hight_freq, hight_freq.size(1) // 3, dim=1)
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
        y = self.act(y)
        return y
    

def conv1x1(in_ch, out_ch):
    return nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=1,
        stride=1,
        padding=0
    )
    

class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvModule(in_ch, in_ch * 2, 3)
        self.act1 = nn.Mish(inplace=True)
        self.conv2 = ConvModule(in_ch * 2, out_ch, 3)

        self.down_bneck = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act_final = nn.Mish(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        y = self.pool(y)
        return y
    

class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvModule(in_ch, in_ch * 2, 3)
        self.act1 = nn.Mish(inplace=True)
        self.conv2 = ConvModule(in_ch * 2, out_ch, 3)

        self.down_bneck = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.act_final = nn.Mish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y


class MiniUNet(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.attn2 = CBAM(mid_ch + mid_ch)
        self.attn1 = CBAM(mid_ch + mid_ch)
        
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.init_block(x)

        down_f1 = self.downsample_block1(hx)
        down_f2 = self.downsample_block2(down_f1)

        deep_f = self.deep_conv_block(down_f2)

        deep_f = self.upsample2(deep_f)
        decoded_f2 = torch.cat((down_f1, deep_f), axis=1)
        decoded_f2, _, sa_attn1 = self.attn2(decoded_f2)
        decoded_f2 = self.upsample_features_block2(decoded_f2)

        down_f1 = self.upsample1(down_f1)
        decoded_f1 = torch.cat((hx, down_f1), dim=1)
        decoded_f1, _, sa_attn2 = self.attn1(decoded_f1)
        decoded_f1 = self.upsample_features_block1(decoded_f1)

        return decoded_f1, [sa_attn1, sa_attn2]


class WTSNet(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()

        out_w_ch = image_channels * 4

        self.dwt1 = DownscaleByWaveletes()
        self.dwt2 = DownscaleByWaveletes()
        self.dwt3 = DownscaleByWaveletes()
        self.dwt4 = DownscaleByWaveletes()

        self.low_freq_to_wavelets_u1 = MiniUNet(image_channels, 64, 64)
        self.hight_freq_u1 = MiniUNet(64 + image_channels * 3, 64, 128)
        self.hight_freq_c1 = conv1x1(128, image_channels * 3)

        self.low_freq_to_wavelets_u2 = MiniUNet(image_channels, 32, 64)
        self.hight_freq_u2 = MiniUNet(64 + image_channels * 3, 64, 64)
        self.hight_freq_c2 = conv1x1(64, image_channels * 3)

        self.low_freq_to_wavelets_u3 = MiniUNet(image_channels, 32, 32)
        self.hight_freq_u3 = MiniUNet(32 + image_channels * 3, 32, 64)
        self.hight_freq_c3 = conv1x1(64, image_channels * 3)

        self.low_freq_u4 = MiniUNet(image_channels, 32, 32)
        self.low_freq_c4 = conv1x1(32, image_channels)
        self.low_freq_to_wavelets_u4 = MiniUNet(image_channels, 32, 32)
        self.hight_freq_u4 = MiniUNet(32 + image_channels * 3, 32, 32)
        self.hight_freq_c4 = conv1x1(32, image_channels * 3)

        self.iwt1 = UpscaleByWaveletes()
        self.iwt2 = UpscaleByWaveletes()
        self.iwt3 = UpscaleByWaveletes()
        self.iwt4 = UpscaleByWaveletes()


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ll1, hf1 = self.dwt1(x)
        ll2, hf2 = self.dwt2(ll1 / 2)
        ll3, hf3 = self.dwt2(ll2 / 2)
        ll4, hf4 = self.dwt2(ll3 / 2)

        t4, _ = self.low_freq_u4(ll4)
        t4_wf, _ = self.low_freq_to_wavelets_u4(ll4)
        df_ll4 = self.low_freq_c4(t4)
        df_hd4, sa4 = self.hight_freq_u4(
            torch.cat((t4_wf, hf4), dim=1)
        )
        df_hd4 = self.hight_freq_c4(df_hd4)
        pred_ll4 = ll4 - df_ll4
        hf4 -= df_hd4

        pred_ll3 = self.iwt4(pred_ll4, hf4)
        t3_wf, _ = self.low_freq_to_wavelets_u3(pred_ll3)
        df_hd3, sa3 = self.hight_freq_u3(
            torch.cat((t3_wf, hf3), dim=1)
        )
        df_hd3 = self.hight_freq_c3(df_hd3)
        hf3 -= df_hd3

        pred_ll2 = self.iwt3(pred_ll3 * 2, hf3)
        t2_wf, _ = self.low_freq_to_wavelets_u2(pred_ll2)
        df_hd2, sa2 = self.hight_freq_u2(
            torch.cat((t2_wf, hf2), dim=1)
        )
        df_hd2 = self.hight_freq_c2(df_hd2)
        hf2 -= df_hd2

        pred_ll1 = self.iwt2(pred_ll2 * 2, hf2)
        t1_wf, _ = self.low_freq_to_wavelets_u1(pred_ll1)
        df_hd1, sa1 = self.hight_freq_u1(
            torch.cat((t1_wf, hf1), dim=1)
        )
        df_hd1 = self.hight_freq_c1(df_hd1)
        hf1 -= df_hd1

        pred_image = self.iwt1(pred_ll1 * 2, hf1)

        sa1 = nn.functional.interpolate(sa1[0], (x.size(2), x.size(3)), mode='area')
        sa2 = nn.functional.interpolate(sa2[0], (x.size(2), x.size(3)), mode='area')
        sa3 = nn.functional.interpolate(sa3[0], (x.size(2), x.size(3)), mode='area')
        sa4 = nn.functional.interpolate(sa4[0], (x.size(2), x.size(3)), mode='area')

        wavelets1 = torch.cat((pred_ll1, hf1), dim=1)
        wavelets2 = torch.cat((pred_ll2, hf2), dim=1)
        wavelets3 = torch.cat((pred_ll3, hf3), dim=1)
        wavelets4 = torch.cat((pred_ll4, hf4), dim=1)

        return [pred_image, wavelets1, wavelets2, wavelets3, wavelets4], [sa1, sa2, sa3, sa4]


if __name__ == '__main__':
    import numpy as np

    device = 'cpu'

    model = WTSNet().to(device)
    inp = torch.rand(1, 3, 512, 512).to(device)

    with torch.no_grad():
        out = model(inp)

    for block_output, block_name in zip(out[0], ('d{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    print()

    for block_output, block_name in zip(out[1], ('sa_{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: {} params'.format(params))
