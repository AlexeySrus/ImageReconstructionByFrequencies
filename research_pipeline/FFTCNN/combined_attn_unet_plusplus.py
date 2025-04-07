from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn

from FFTCNN.combined_attn_unet import FeaturesDownsample, FeaturesUpsample, conv3x3, conv1x1
from FFTCNN.attention import FFTCAFSModule
from utils.resample import resample_lanczos


class RestrictedFeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int, use_attention: bool = True):
        super().__init__()

        mid_ch = max(out_ch, 16)

        self.init_conv = conv3x3(in_ch, mid_ch)

        self.use_attention = use_attention
        if use_attention:
            self.attn1 = FFTCAFSModule(channel=mid_ch, reduction=32, image_size=image_size)
        else:
            self.attn1 = None

        self.conv1 = conv3x3(mid_ch, mid_ch * 2)
        self.norm1 = nn.BatchNorm2d(mid_ch * 2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(mid_ch * 2, out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

        self.down_bneck = conv1x1(mid_ch, out_ch)

        self.act_final = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.init_conv(x)

        if self.use_attention:
            y, sa_1 = self.attn1(hx)
        else:
            y = hx
            sa_1 = []

        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y, sa_1


class CompactFeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()

        self.up = lambda x: resample_lanczos(x, scale=2, align_corners=False)
        # self.features = FeaturesProcessing(in_ch, out_ch, window_size=window_size, image_size=image_size, use_attention=False) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        # y, _ = self.features(y)
        return y, []


class FFTAttentionUNetPlusPlusModule(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, need_up_features: bool = False, image_size: int = 256):
        super().__init__()

        self.init_block = RestrictedFeaturesProcessing(in_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)

        self.init_block_with_attn = RestrictedFeaturesProcessing(mid_ch, mid_ch, window_size=64, image_size=image_size)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch, window_size=64, image_size=image_size)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2, window_size=32, image_size=image_size // 2)
        self.downsample_block3 = FeaturesDownsample(mid_ch * 2, mid_ch * 3, window_size=16, image_size=image_size // 4)
        self.downsample_block4 = FeaturesDownsample(mid_ch * 3, mid_ch * 4, window_size=8, image_size=image_size // 8)

        self.deep_conv_block = RestrictedFeaturesProcessing(mid_ch * 4, mid_ch * 4, window_size=8, image_size=image_size // 16)

        upsample_module = FeaturesUpsample

        self.middle_upsample_3_2 = CompactFeaturesUpsample(mid_ch * 3, mid_ch * 3, window_size=16, image_size=image_size // 4)
        self.middle_process_2_4 = RestrictedFeaturesProcessing(mid_ch * 3 + mid_ch * 2, mid_ch * 2, window_size=16, image_size=image_size // 4, use_attention=False)

        self.middle_upsample_2_1_3 = CompactFeaturesUpsample(mid_ch * 2, mid_ch * 2, window_size=32, image_size=image_size // 2)
        self.middle_upsample_2_1_4 = CompactFeaturesUpsample(mid_ch * 2, mid_ch * 2, window_size=32, image_size=image_size // 2)
        self.middle_process_1_3 = RestrictedFeaturesProcessing(mid_ch + mid_ch * 2, mid_ch, window_size=32, image_size=image_size // 2, use_attention=False)
        self.middle_process_1_5 = RestrictedFeaturesProcessing(mid_ch + mid_ch + mid_ch * 2, mid_ch, window_size=32, image_size=image_size // 2, use_attention=False)

        self.middle_upsample_1_0_2 = CompactFeaturesUpsample(mid_ch, mid_ch, window_size=64, image_size=image_size)
        self.middle_upsample_1_0_4 = CompactFeaturesUpsample(mid_ch, mid_ch, window_size=64, image_size=image_size)
        self.middle_upsample_1_0_6 = CompactFeaturesUpsample(mid_ch, mid_ch, window_size=64, image_size=image_size)
        self.middle_process_0_2 = RestrictedFeaturesProcessing(mid_ch + mid_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)
        self.middle_process_0_4 = RestrictedFeaturesProcessing(mid_ch + mid_ch + mid_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)
        self.middle_process_0_6 = RestrictedFeaturesProcessing(mid_ch + mid_ch + mid_ch + mid_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)

        self.upsample4 = upsample_module(mid_ch * 4, mid_ch * 3, window_size=16, image_size=image_size // 8)
        self.upsample3 = upsample_module(mid_ch * 3, mid_ch * 2, window_size=32, image_size=image_size // 4)
        self.upsample2 = upsample_module(mid_ch * 2, mid_ch, window_size=64 , image_size=image_size // 2)
        self.upsample1 = upsample_module(mid_ch, mid_ch, window_size=64 , image_size=image_size)
        
        self.upsample_features_block4 = RestrictedFeaturesProcessing(mid_ch * 3 + mid_ch * 3, mid_ch * 3, window_size=8, image_size=image_size // 8)
        self.upsample_features_block3 = RestrictedFeaturesProcessing(mid_ch * 2 + mid_ch * 2 + mid_ch * 2, mid_ch * 2, window_size=16, image_size=image_size // 4)
        self.upsample_features_block2 = RestrictedFeaturesProcessing(mid_ch + mid_ch + mid_ch, mid_ch, window_size=32, image_size=image_size // 2)
        self.upsample_features_block1 = RestrictedFeaturesProcessing(mid_ch + mid_ch + mid_ch + mid_ch + mid_ch, out_ch, window_size=64 , image_size=image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx, _ = self.init_block(x)
        hx, sa_init = self.init_block_with_attn(hx)

        down_f1, sa_f1 = self.downsample_block1(hx)         # W // 2
        down_f2, sa_f2 = self.downsample_block2(down_f1)    # W // 4
        down_f3, sa_f3 = self.downsample_block3(down_f2)    # W // 8
        down_f4, sa_f4 = self.downsample_block4(down_f3)    # W // 16

        deep_f, sa_f = self.deep_conv_block(down_f4)

        deep_f, sa_up_4 = self.upsample4(deep_f)
        decoded_f4 = torch.cat((down_f3, deep_f), axis=1)
        decoded_f4, sa_df4 = self.upsample_features_block4(decoded_f4)

        plus_plus_up_3_2, _ = self.middle_upsample_3_2(down_f3)
        plus_plus_f_2_4, _ = self.middle_process_2_4(torch.cat((down_f2, plus_plus_up_3_2), axis=1))

        deep_f, sa_up_3 = self.upsample3(decoded_f4)
        decoded_f3 = torch.cat((down_f2, plus_plus_f_2_4, deep_f), axis=1)
        decoded_f3, sa_df3 = self.upsample_features_block3(decoded_f3)

        plus_plus_up_2_1_2, _ = self.middle_upsample_2_1_3(down_f2)
        plus_plus_f_1_3, _ = self.middle_process_1_3(torch.cat((down_f1, plus_plus_up_2_1_2), axis=1))

        plus_plus_up_2_1_4, _ = self.middle_upsample_2_1_4(plus_plus_f_2_4)
        plus_plus_f_1_5, _ = self.middle_process_1_5(torch.cat((down_f1, plus_plus_f_1_3, plus_plus_up_2_1_4), axis=1))

        deep_f, sa_up_2 = self.upsample2(decoded_f3)
        decoded_f2 = torch.cat((down_f1, plus_plus_f_1_5, deep_f), axis=1)
        decoded_f2, sa_df2 = self.upsample_features_block2(decoded_f2)

        plus_plus_up_1_0_2, _ = self.middle_upsample_1_0_2(down_f1)
        plus_plus_f_0_2, _ = self.middle_process_0_2(torch.cat((hx, plus_plus_up_1_0_2), axis=1))

        plus_plus_up_1_0_4,_  = self.middle_upsample_1_0_4(plus_plus_f_1_3)
        plus_plus_f_0_4, _ = self.middle_process_0_4(torch.cat((hx, plus_plus_f_0_2, plus_plus_up_1_0_4), axis=1))

        plus_plus_up_1_0_6, _ = self.middle_upsample_1_0_6(plus_plus_f_1_5)
        plus_plus_f_0_6, _ = self.middle_process_0_6(torch.cat((hx, plus_plus_f_0_2, plus_plus_f_0_4, plus_plus_up_1_0_6), axis=1))

        deep_f, sa_up_1 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, plus_plus_f_0_2, plus_plus_f_0_4, plus_plus_f_0_6, deep_f), dim=1)
        decoded_f1, sa_df1 = self.upsample_features_block1(decoded_f1)

        decoded_series = [decoded_f1, plus_plus_f_0_6, plus_plus_f_0_4, plus_plus_f_0_2]

        return decoded_series, sa_init + sa_f1 + sa_f2 + sa_f3 + sa_f4 + sa_f + sa_up_4 + sa_up_3 + sa_up_2 + sa_up_1 + sa_df4 + sa_df3 + sa_df2 + sa_df1


class FFTAttentionUNetPlusPlus(nn.Module):
    def __init__(self, in_ch: int = 3,  out_ch: int = 3, image_size: int = 256, use_substraction: bool = False):
        super().__init__()

        mid_ch = 16

        self.unet = FFTAttentionUNetPlusPlusModule(in_ch, mid_ch, out_ch, image_size=image_size)
        self.out_convs = nn.ModuleList(
            [
                nn.Conv2d(out_ch if i == 0 else mid_ch, out_ch, 1, bias=True)
                for i in range(4)
            ]
        )
        self.export = False
        self.use_substraction = use_substraction

    def to_export(self):
        self.export = True

    def norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1

    def denorm_input(self, x: torch.Tensor) -> torch.Tensor:
        return [(x[out_i] + 1) * 0.5 for out_i in range(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.norm_input(x)

        y, sa_list = self.unet(hx)
        y = [self.out_convs[out_i](y[out_i]) for out_i in range(4)]

        if self.export:
            if self.use_substraction:
                return self.denorm_input([hx + y[out_i] for out_i in range(4)])
            return self.denorm_input(y)

        if self.training:
            with torch.no_grad():
                sa_list = [
                    nn.functional.interpolate(torch.abs(sa), (x.size(2), x.size(3)), mode='bilinear')
                    for sa in sa_list
                ]

        if self.use_substraction:
            return self.denorm_input([hx + y[out_i] for out_i in range(4)]), sa_list
        
        return self.denorm_input(y), sa_list

if __name__ == '__main__':
    model = FFTAttentionUNetPlusPlus()

    t = torch.rand(1, 3, 256, 256)
    out = model(t)

    res = out[0]

    print(res[0].detach().shape)
