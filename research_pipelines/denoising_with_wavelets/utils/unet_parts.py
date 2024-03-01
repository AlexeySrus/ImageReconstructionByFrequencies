# From original repository: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
""" Parts of the U-Net model """

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from WTSNet.attention import CBAM


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvWithoutLastActivation(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in1_channels, in2_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in1_channels + in2_channels, out_channels, in1_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in1_channels, in1_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in1_channels // 2 + in2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class OneLevelUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bilinear: bool = False) -> None:
        super().__init__()

        self.init_feats = DoubleConvWithoutLastActivation(in_ch, out_ch * 8)

        if use_bilinear:
            self.bottleneck_feats_layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=0.25),
                DoubleConv(out_ch * 8, out_ch * 16),
                nn.UpsamplingBilinear2d(scale_factor=4),
                DoubleConvWithoutLastActivation(out_ch * 16, out_ch * 8)
            )
        else:
            self.bottleneck_feats_layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=4, stride=4),
                DoubleConv(out_ch * 8, out_ch * 16),
                nn.ConvTranspose2d(out_ch * 16, out_ch * 8, kernel_size=4, stride=4),
                DoubleConvWithoutLastActivation(out_ch * 8, out_ch * 8)
            )

        self.up_cbam = CBAM(out_ch * 8, reduction=min(out_ch * 8, 16))
        self.up_layer = DoubleConv(out_ch * 8, out_ch)

        self.out_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        in_feats = self.init_feats(2 * x - 1) # x in 0..1 -> -1..1
        bottleneck_features = self.bottleneck_feats_layer(in_feats)
        # x_up = torch.concatenate((in_feats, bottleneck_features), dim=1)
        x_up = nn.functional.relu(in_feats + bottleneck_features)
        x_up, _, up_sa = self.up_cbam(x_up)
        res_feats = self.up_layer(x_up)
        out = self.out_conv(res_feats)
        return out / 2 + 0.5, up_sa    # out in -1..1 -> 0..1


if __name__ == '__main__':
    net = OneLevelUNet(3, 1)

    x = torch.rand(1, 3, 224, 224)

    pamars_count = sum(p.numel() for p in net.parameters())
    print('Parameters count: {:.1f} K'.format(pamars_count / 1E+3))

    out, _ = net(x)
    print(out.shape)
