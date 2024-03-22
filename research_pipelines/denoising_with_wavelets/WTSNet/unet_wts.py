from typing import Iterator, Tuple, List, Optional, Callable
from collections import OrderedDict
import kornia
import torch
import torch.nn as nn
from timm import create_model


from WTSNet.attention import CBAM, CoordinateAttention
from WTSNet.wtsnet import UpscaleByWaveletes, DownscaleByWaveletes, conv1x1, conv3x3, FeaturesProcessingWithLastConv
from utils.haar_utils import HaarForward, HaarInverse
from utils.unet_parts import Up as UNetUp, OneLevelUNet


padding_mode: str = 'reflect'
activation_function_constructor = nn.ReLU


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: torch.nn.Module = nn.ReLU, use_lact_act: bool = False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.act1 = act()
        self.conv2 = conv3x3(in_ch * 2, out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = act() if use_lact_act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)

        hx = self.down_bneck(hx)

        y = hx + y

        if self.act_final is not None:
            y = self.act_final(y)

        return y



class FeaturesDownSampleProcessing(nn.Module):
    def __init__(self, ch: int, act: torch.nn.Module = nn.ReLU):
        super().__init__()
        
        assert ch > 0

        self.wavelet_pool = HaarForward()
        self.features = FeaturesProcessing(ch * 4, ch * 2, act)
        self.last_conv = conv1x1(ch * 2, ch * 2)

    def forward(self, x):
        y = self.wavelet_pool(x)
        y = self.features(y)
        y = self.last_conv(y)
        return y


class FeaturesUpSampleProcessing(nn.Module):
    def __init__(self, ch: int, act: torch.nn.Module = nn.ReLU):
        super().__init__()
        
        assert ch > 0

        self.features = FeaturesProcessing(ch, ch * 2, act, use_lact_act=False)
        self.wavelet_unpool = HaarInverse()

    def forward(self, x):
        y = self.features(x)
        y = self.wavelet_unpool(y)
        return y


class WaveletEncoder(nn.Module):
    def __init__(self, in_ch: int, start_ch: int, act: torch.nn.Module = nn.ReLU, depth: int = 5):
        super().__init__()

        self.init_conv = nn.Conv2d(in_ch, start_ch, kernel_size=1)

        self.enc_channels = [
            start_ch * (2 ** (i + 1))
            for i in range(depth)
        ]

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    FeaturesProcessing(start_ch * (2 ** i), start_ch * (2 ** i), act),
                    FeaturesDownSampleProcessing(start_ch * (2 ** i), act),
                )
                for i in range(depth)
            ]
        )

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        y = self.init_conv(input_tensor)
        out_feats = []
        for layer in self.layers:
            y = layer(y)
            out_feats.append(y)

        return out_feats


class MWCNNCBAM(nn.Module):
    def __init__(self, depth: int = 5, for_wavelet_loss: bool = False, act=nn.ReLU):
        super().__init__()

        in_ch: int = 3
        mid_ch: int = 3
        out_ch: int = 3

        self.encoder_model = WaveletEncoder(in_ch, mid_ch, act=activation_function_constructor, depth=depth)
        self.enc_channels = self.encoder_model.enc_channels

        self.attn_layers = nn.ModuleList(
            [
                CBAM(
                    channel=2*in_ch * (2 ** (i + 1)) if i < depth - 1 else in_ch * (2 ** (i + 1)), 
                    reduction=min(in_ch * (2 ** (i + 1)), 16),
                    kernel_size=7 if i < 3 else 3
                )
                for i in range(depth)
            ]
        )

        self.inter_features_layers = nn.ModuleList(
            [
                FeaturesProcessing(
                    in_ch=2*in_ch * (2 ** (i + 1)) if i < depth - 1 else in_ch * (2 ** (i + 1)),
                    out_ch=in_ch * (2 ** (i + 1)), 
                    act=act, use_lact_act=False
                )
                for i in range(depth)
            ]
        )

        self.decoder_layers = nn.ModuleList(
            [
                FeaturesUpSampleProcessing(
                    ch=in_ch * (2 ** (i + 1)), 
                    act=act
                )
                for i in range(depth)
            ]
        )

        self.wavelet_projection_layers = None

        if for_wavelet_loss:
            self.wavelet_projection_layers = nn.ModuleList(
                [
                    conv3x3(in_ch=in_ch * (2 ** (i + 1)), out_ch=4*3)
                    for i in range(depth)
                ]
            )

        self.last_conv = conv1x1(mid_ch, out_ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        encoder_features = self.encoder_model(x)

        attn_maps = []
        wavelets_maps = []

        for i in range(len(encoder_features) - 1, -1, -1):
            if i < len(encoder_features) - 1:
                y = torch.cat([encoder_features[i], y], dim=1)
            else:
                y = encoder_features[-1]

            y, _, sa = self.attn_layers[i](y)
            attn_maps.append(sa)

            inter_feats = self.inter_features_layers[i](y)

            if self.wavelet_projection_layers is not None:
                wavelets_out = self.wavelet_projection_layers[i](inter_feats)
                wavelets_maps.append(wavelets_out)

            y = self.decoder_layers[i](inter_feats)

        y = self.last_conv(y)

        with torch.no_grad():
            for i in range(len(encoder_features)):
                attn_maps[i] = nn.functional.interpolate(attn_maps[i], (x.size(2), x.size(3)), mode='area')

        wavelets_maps.reverse()
        attn_maps.reverse()

        return y, wavelets_maps, attn_maps



if __name__ == '__main__':
    model = MWCNNCBAM(5, for_wavelet_loss=True)

    pamars_count = sum(p.numel() for p in model.parameters())
    print('Parameters count: {:.2f} M'.format(pamars_count / 1E+6))

    print(model.enc_channels)

    x = torch.rand(1, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    print('Out:', out[0].shape)

    for w in out[1]:
        print(w.shape)