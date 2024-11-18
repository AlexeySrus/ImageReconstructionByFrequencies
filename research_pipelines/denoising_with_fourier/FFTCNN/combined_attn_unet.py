from typing import Tuple, List, Optional
from collections import OrderedDict
import math
import torch
import torch.nn as nn

from FFTCNN.attention import FFTCAFSModule, SpatialAttention, ChannelAttention

from FFTCNN.interpolation_type import DownSampleMode, UpSampleMode, InterpolationMode, \
        get_down_function, get_up_function

import os
DEPRECATED_IMPLEMENTATION: Optional[str] = os.getenv("DEPRECATED_IMPLEMENTATION")

padding_mode: str = 'reflect'



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def init_weights_kaiming(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)


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
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn1 = FFTCAFSModule(channel=in_ch, reduction=32, image_size=image_size)
        else:
            self.attn1 = None

        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.norm1 = nn.BatchNorm2d(in_ch * 2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(in_ch * 2, out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        if self.use_attention:
            y, sa_1 = self.attn1(x)
        else:
            y = x
            sa_1 = []

        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        
        return y, sa_1


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int, use_attention: bool = True, interpolation_mode: DownSampleMode = DownSampleMode.MAXPOOL):
        super().__init__()
        self.features_in = FeaturesProcessing(in_ch, in_ch * 2, window_size=window_size, image_size=image_size, use_attention=use_attention)
        self.pool = get_down_function(interpolation_mode)
        self.features_out = FeaturesProcessing(in_ch * 2, out_ch, window_size=window_size, image_size=image_size, use_attention=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.features_in(x)
        y = self.pool(y)
        y, _ = self.features_out(y)
        return y, sa


class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int, use_attention: bool = True, interpolation_mode: UpSampleMode = UpSampleMode.BILINEAR):
        super().__init__()
        self.in_features = FeaturesProcessing(in_ch, in_ch, window_size=window_size, image_size=image_size, use_attention=use_attention)
        if DEPRECATED_IMPLEMENTATION is None:
            self.up = get_up_function(interpolation_mode, in_ch)
        else:
            print('Warning: Use deprecated UP method in FeaturesUpsample')
            self.up = lambda x: torch.nn.functional.interpolate(x, scale_factor=2, align_corners=True, mode='bilinear')
        self.features = FeaturesProcessing(in_ch, out_ch, window_size=window_size, image_size=image_size, use_attention=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.in_features(x)
        y = self.up(y)
        y, _ = self.features(y)
        return y, sa


class FFTAttentionUNetModule(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, image_size: int = 256, 
                 attention_mode: str = 'full', interolation_mode: InterpolationMode = InterpolationMode.MAXPOOL_BILINEAR):
        super().__init__()
        down_interpolation_mode = interolation_mode.value[0]
        up_interpolation_mode = interolation_mode.value[1]

        self.init_block = FeaturesProcessing(in_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)

        self.init_block_2 = FeaturesProcessing(mid_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch, window_size=64, image_size=image_size, use_attention=False, interpolation_mode=down_interpolation_mode)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2, window_size=32, image_size=image_size // 2, use_attention=False, interpolation_mode=down_interpolation_mode)
        self.downsample_block3 = FeaturesDownsample(mid_ch * 2, mid_ch * 3, window_size=16, image_size=image_size // 4, use_attention=False, interpolation_mode=down_interpolation_mode)
        self.downsample_block4 = FeaturesDownsample(mid_ch * 3, mid_ch * 4, window_size=8, image_size=image_size // 8, use_attention=False, interpolation_mode=down_interpolation_mode)

        self.connection_attn1 = FFTCAFSModule(channel=mid_ch, reduction=16, image_size=image_size, mode=attention_mode)
        self.connection_attn2 = FFTCAFSModule(channel=mid_ch, reduction=16, image_size=image_size // 2, mode=attention_mode)
        self.connection_attn3 = FFTCAFSModule(channel=mid_ch * 2, reduction=32, image_size=image_size // 4, mode=attention_mode)
        self.connection_attn4 = FFTCAFSModule(channel=mid_ch * 3, reduction=32, image_size=image_size // 8, mode=attention_mode)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 4, mid_ch * 4, window_size=8, image_size=image_size // 16, use_attention=False)

        upsample_module = FeaturesUpsample

        self.upsample4 = upsample_module(mid_ch * 4, mid_ch * 3, window_size=16, image_size=image_size // 8, use_attention=False, interpolation_mode=up_interpolation_mode)
        self.upsample3 = upsample_module(mid_ch * 3, mid_ch * 2, window_size=32, image_size=image_size // 4, use_attention=False, interpolation_mode=up_interpolation_mode)
        self.upsample2 = upsample_module(mid_ch * 2, mid_ch, window_size=64 , image_size=image_size // 2, use_attention=False, interpolation_mode=up_interpolation_mode)
        self.upsample1 = upsample_module(mid_ch, mid_ch, window_size=64 , image_size=image_size, use_attention=False, interpolation_mode=up_interpolation_mode)
        
        self.upsample_features_block4 = FeaturesProcessing(mid_ch * 3 + mid_ch * 3, mid_ch * 3, window_size=8, image_size=image_size // 8, use_attention=False)
        self.upsample_features_block3 = FeaturesProcessing(mid_ch * 2 + mid_ch * 2, mid_ch * 2, window_size=16, image_size=image_size // 4, use_attention=False)
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch, window_size=32, image_size=image_size // 2, use_attention=False)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch, window_size=64 , image_size=image_size, use_attention=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx, _ = self.init_block(x)
        hx, _ = self.init_block_2(hx)

        down_f1, _ = self.downsample_block1(hx)         # W // 2
        down_f2, _ = self.downsample_block2(down_f1)    # W // 4
        down_f3, _ = self.downsample_block3(down_f2)    # W // 8
        down_f4, _ = self.downsample_block4(down_f3)    # W // 16

        deep_f, _ = self.deep_conv_block(down_f4)

        hx,      sa_f1 = self.connection_attn1(hx)
        down_f1, sa_f2 = self.connection_attn2(down_f1)
        down_f2, sa_f3 = self.connection_attn3(down_f2)
        down_f3, sa_f4 = self.connection_attn4(down_f3)

        deep_f, _ = self.upsample4(deep_f)
        decoded_f4 = torch.cat((down_f3, deep_f), axis=1)
        decoded_f4, _ = self.upsample_features_block4(decoded_f4)

        deep_f, _ = self.upsample3(decoded_f4)
        decoded_f3 = torch.cat((down_f2, deep_f), axis=1)
        decoded_f3, _ = self.upsample_features_block3(decoded_f3)

        deep_f, _ = self.upsample2(decoded_f3)
        decoded_f2 = torch.cat((down_f1, deep_f), axis=1)
        decoded_f2, _ = self.upsample_features_block2(decoded_f2)

        deep_f, _ = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, deep_f), dim=1)
        decoded_f1, _ = self.upsample_features_block1(decoded_f1)

        return decoded_f1, sa_f1 + sa_f2 + sa_f3 + sa_f4


class FFTAttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3,  out_ch: int = 3, image_size: int = 256, use_substraction: bool = False, 
                 attention_mode: str = 'full', interolation_mode: InterpolationMode = InterpolationMode.MAXPOOL_BILINEAR):
        super().__init__()

        self.unet = FFTAttentionUNetModule(in_ch, 32, out_ch, image_size=image_size, attention_mode=attention_mode, interolation_mode=interolation_mode)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, bias=True)
        self.export = False
        self.use_substraction = use_substraction

    def to_export(self):
        self.export = True

    def norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1

    def denorm_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.norm_input(x)

        y, sa_list = self.unet(hx)
        y = self.out_conv(y)

        if self.export:
            if self.use_substraction:
                return self.denorm_input(hx + y)
            return self.denorm_input(y)

        # if self.training:
        with torch.no_grad():
            sa_list = [
                nn.functional.interpolate(torch.abs(sa), (x.size(2), x.size(3)), mode='bilinear')
                for sa in sa_list
            ]

        if self.use_substraction:
            return self.denorm_input(hx + y), sa_list
        
        return self.denorm_input(y), sa_list
    
    def custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.norm_input(x)
        y, sa_list = self.unet(hx)
        y = self.out_conv(y)

        sa_list = [
            nn.functional.interpolate(torch.abs(sa), (x.size(2), x.size(3)), mode='bilinear')
            for sa in sa_list
        ]

        res = torch.concat(sa_list, dim=1)
        return res


if __name__ == '__main__':
    import cv2
    import numpy as np
    from timeit import default_timer as time
    import segmentation_models_pytorch as smp


    model = FFTAttentionUNet(3, 3, use_substraction=True, interolation_mode=InterpolationMode.MAXPOOL_BILINEAR, attention_mode='none')

    m_params = sum(p.numel() for p in model.parameters())

    print('Ours model params: {}M'.format(m_params // 10 ** 6))

    # unet_model = smp.Unet(
    #     encoder_name="resnet18",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=3,
    # )
    unet_model = model
    unet_model_params = sum(p.numel() for p in unet_model.parameters())

    print('UNet model params: {}M'.format(unet_model_params // 10 ** 6))

    t = torch.rand(1, 3, 256, 256)
    out = model(t)
    sa_list = out[1]

    sa_tensors = []
    for k in range(len(sa_list) // 4):
        sa_tensors.append(torch.concat([sa_list[2*k + q] for q in range(4)], dim=3))

    sa_tensor = torch.concat(sa_tensors, dim=2)
    print(sa_tensor.shape)

    print(out[0].shape)

    _ = model.eval()
    model.to_export()

    with torch.no_grad():
        out = model(t)

    n_attempts: int = 100

    start_time = time()
    with torch.no_grad():
        for _ in range(n_attempts):
            out = model(t)
    finish_time = time()

    infer_time = (finish_time - start_time) / n_attempts
    print('Inference time: {:.5f} sec'.format(infer_time))

    # model.to_export()
    # model.eval()

    # traced = torch.jit.trace(model, example_inputs=t)
    # torch.jit.save(traced, '/home/alexey/Downloads/fftcnn.pt')
