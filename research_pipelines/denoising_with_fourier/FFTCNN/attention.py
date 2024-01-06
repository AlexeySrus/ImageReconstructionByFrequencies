""" 
PyTorch implementation of CBAM: Convolutional Block Attention Module

As described in https://arxiv.org/pdf/1807.06521

The attention mechanism is achieved by using two different types of attention gates: 
channel-wise attention and spatial attention. The channel-wise attention gate is applied 
to each channel of the input feature map, and it allows the network to focus on the most 
important channels based on their spatial relationships. The spatial attention gate is applied 
to the entire input feature map, and it allows the network to focus on the most important regions 
of the image based on their channel relationships.
"""

from typing import Tuple

import numpy as np
import math
import torch
from torch import nn

from FFTCNN.mixvit import LayerNorm, RISwish, OverlapPatchEmbed, Block


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                    y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel


def real_imaginary_leakyrelu(z):
    return nn.functional.leaky_relu(z.real) + 1.j * nn.functional.leaky_relu(z.imag)


class RealImaginaryLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return real_imaginary_leakyrelu(z) 
    

class FFTMaxPool2D(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, z, return_indices: bool = False):
        _, max_indices = torch.nn.functional.max_pool2d_with_indices(torch.abs(z), self.kernel_size, self.stride)
        max_out = retrieve_elements_from_indices(z, max_indices)

        if return_indices:
            return max_out, max_indices

        return max_out
    

class Unet1lvl(nn.Module):
    padding_mode = 'reflect'
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Unet1lvl, self).__init__()

        self.process1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.process2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU()
        )

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(mid_ch // 2),
            nn.LeakyReLU()
        )

        self.process3 = nn.Sequential(
            nn.Conv2d(mid_ch // 2 + mid_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

        self.last_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.process1(x)
        yp1 = self.pool(y)
        ydf1 = self.process2(yp1)
        yup1 = self.up(ydf1)
        yup1 = torch.cat((yup1, y), dim=1)
        out = self.process3(yup1)
        out = self.last_conv(out)
        return out


class FullComplexSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(FullComplexSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, dtype=torch.cfloat, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        avg_out = torch.mean(z, dim=1, keepdim=True)
        _, max_indices = torch.max(torch.abs(z), dim=1, keepdim=True)
        max_out = retrieve_elements_from_indices(z, max_indices)
        
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return z * attn, attn


class ComplexSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ComplexSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = torch.abs(z)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return z * attn.to(torch.cfloat), attn


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False, padding_mode='reflect')
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        return x * attn, attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return x * attn, attn


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x, ca_tensor = self.ca(x)
        x, sa_tensor = self.sa(x)
        return x, ca_tensor, sa_tensor


class ComplexSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(ComplexSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.key = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.value = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(torch.abs(scores)).to(torch.cfloat)
        weighted = torch.bmm(attention, values)
        return weighted


def real_imaginary_leaky_relu(z):
    return nn.functional.leaky_relu(z.real) + 1.j * nn.functional.leaky_relu(z.imag)


class ComplexAttnMLP(nn.Module):
    def __init__(self, in_feats: int, mid_feats: int, out_feats: int):
            super(ComplexAttnMLP, self).__init__()

            self.layer1 = nn.Linear(in_feats, mid_feats, dtype=torch.cfloat)
            self.self_attn = ComplexSelfAttention(mid_feats)
            self.layer2 = nn.Linear(mid_feats, out_feats, dtype=torch.cfloat)

    def forward(self, x):
        y = self.layer1(x)
        y = real_imaginary_leaky_relu(y)
        y = self.self_attn(y)
        y = self.layer2(y)
        return y
    
class MixVitAttention(nn.Module):
    def __init__(self, in_ch: int, patch_size: int, image_size: int) -> None:
        super().__init__()
        self.patch_embedder = OverlapPatchEmbed(image_size, patch_size, in_chans=in_ch, stride=4, embed_dim=16)
        self.block = Block(16, 1)

        self.pred_filter = nn.Sequential(
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        p, H, W = self.patch_embedder(z)
        out = self.block(p, H, W)
        out = out.reshape(z.size(0), H, W, -1).permute(0, 3, 1, 2).contiguous()
        attn = nn.functional.interpolate(torch.abs(out), (z.size(2), z.size(3)), mode='bilinear')
        attn = self.pred_filter(attn)
        return z * attn.to(torch.cfloat), attn


class WindowBasedSelfAttention(nn.Module):
    def __init__(self, window_size:int = 64):
        super(WindowBasedSelfAttention, self).__init__()
        self.mlp = ComplexAttnMLP(window_size ** 2, window_size ** 2 // 4, window_size ** 2)
        self.wsize = window_size

    def forward(self, x):
        features_folds = x.unfold(
            2, size=self.wsize, step=self.wsize).unfold(
                3, size=self.wsize, step=self.wsize).contiguous()

        four_folds = torch.fft.fft2(features_folds, norm='ortho')
        four_folds = torch.fft.fftshift(four_folds)

        _, max_indices = torch.max(torch.abs(four_folds), dim=1, keepdim=True)
        folds = retrieve_elements_from_indices(four_folds, max_indices)
        # folds = torch.mean(four_folds, dim=1, keepdim=True)
        # folds = four_folds

        init_folds_shape = folds.shape

        folds = folds.view(
            folds.size(0), 
            folds.size(1) * folds.size(2) * folds.size(3),
            folds.size(4) * folds.size(5)
        )

        # fft_filter = torch.sigmoid(self.mlp(folds))

        # fft_filter = fft_filter.view(*init_folds_shape)
        # out = four_folds * fft_filter
        
        attn = self.mlp(folds)
        attn = attn.view(*init_folds_shape)
        out = four_folds * attn

        with torch.no_grad():
            fft_filter = torch.clone(attn.mean(dim=1).unsqueeze(1))

        out = torch.fft.ifftshift(out)
        out = torch.fft.ifft2(out, norm='ortho').real

        # corners = out[:, :, ::2, ::2]
        # anchors = out[:, :, 1::2, 1::2]
        # path_size = self.wsize

        # corners[:, :, :-1, :-1, path_size//2:, path_size//2:] = \
        #                                 corners[:, :, :-1, :-1, path_size//2:, path_size//2:] / 2 + \
        #                                 anchors[:, :, :, :, :path_size//2, :path_size//2] / 2

        # corners[:, :, :-1, 1:, path_size//2:, :path_size//2] = \
        #                                         corners[:, :, :-1, 1:, path_size//2:, :path_size//2] / 2 + \
        #                                         anchors[:, :, :, :, :path_size//2, path_size//2:] / 2

        # corners[:, :, 1:, :-1, :path_size//2, path_size//2:] = \
        #                                         corners[:, :, 1:, :-1, :path_size//2, path_size//2:] / 2 + \
        #                                         anchors[:, :, :, :, path_size//2:, :path_size//2] / 2

        # corners[:, :, 1:, 1:, :path_size//2, :path_size//2] = \
        #                                         corners[:, :, 1:, 1:, :path_size//2, :path_size//2] / 2 + \
        #                                         anchors[:, :, :, :, path_size//2:, path_size//2:] / 2


        # out = corners
        out = torch.cat([out[:, :, :, i] for i in range(out.size(3))], dim=4)
        out = torch.cat([out[:, :, i] for i in range(out.size(2))], dim=2)

        with torch.no_grad():
            fft_filter = torch.cat([fft_filter[:, :, :, i] for i in range(fft_filter.size(3))], dim=4)
            fft_filter = torch.cat([fft_filter[:, :, i] for i in range(fft_filter.size(2))], dim=2)

        return out, fft_filter
    

class LowHightFrequencyImageComponents(nn.Module):
    def __init__(self, shape: Tuple[int, int]):
        super().__init__()

        hight_pass_kernel = 1.0 - generate_batt(shape, 500, 1).astype(np.float32)
        low_pass_kernel = generate_batt(shape, 500, 1).astype(np.float32)

        hight_pass_kernel = torch.from_numpy(hight_pass_kernel).unsqueeze(0).unsqueeze(0)
        hight_pass_kernel = hight_pass_kernel.to(torch.cfloat)
        
        low_pass_kernel = torch.from_numpy(low_pass_kernel).unsqueeze(0).unsqueeze(0)
        low_pass_kernel = low_pass_kernel.to(torch.cfloat)
        
        self.hight_pass_kernel = nn.Parameter(hight_pass_kernel, requires_grad=False)
        self.low_pass_kernel = nn.Parameter(low_pass_kernel, requires_grad=False)

    def apply_fft_kernel(self, x, kernel):
        return x * kernel

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.fft.fft2(x, norm='ortho')
        z = torch.fft.fftshift(z)
        
        hight_freq_z = self.apply_fft_kernel(z, self.hight_pass_kernel)
        low_freq_z = self.apply_fft_kernel(z, self.low_pass_kernel)
        
        hight_freq_z = torch.fft.ifftshift(hight_freq_z)
        low_freq_z = torch.fft.ifftshift(low_freq_z)
        
        hight_freq_x = torch.fft.ifft2(hight_freq_z, norm='ortho').real
        low_freq_x = torch.fft.ifft2(low_freq_z, norm='ortho').real
        
        return low_freq_x, hight_freq_x
    

class HightFrequencyImageComponents(nn.Module):
    def __init__(self, shape: Tuple[int, int]):
        super().__init__()

        hight_pass_kernel = 1.0 - generate_batt(shape, 500, 1).astype(np.float32)

        hight_pass_kernel = torch.from_numpy(hight_pass_kernel).unsqueeze(0).unsqueeze(0)
        hight_pass_kernel = hight_pass_kernel.to(torch.cfloat)
        
        self.hight_pass_kernel = nn.Parameter(hight_pass_kernel, requires_grad=False)

    def apply_fft_kernel(self, x, kernel):
        return x * kernel

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.fft.fft2(x, norm='ortho')
        z = torch.fft.fftshift(z)
        
        hight_freq_z = self.apply_fft_kernel(z, self.hight_pass_kernel)
        
        hight_freq_z = torch.fft.ifftshift(hight_freq_z)
        
        hight_freq_x = torch.fft.ifft2(hight_freq_z, norm='ortho').real
        
        return hight_freq_x


class FrequencySplitSpatialAttention(nn.Module):
    def __init__(self, channel: int, image_size: int, kernel_size=7):
        super(FrequencySplitSpatialAttention, self).__init__()

        self.freq_slitter = LowHightFrequencyImageComponents((image_size, image_size))

        self.hlf = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel)
        )
        self.llf = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        low_freq_features, hight_freq_features = self.freq_slitter(x)

        low_freq_features = self.llf(low_freq_features)
        hight_freq_features = self.hlf(hight_freq_features)

        united_features = low_freq_features + hight_freq_features
        united_features = nn.functional.leaky_relu(united_features)

        avg_out = torch.mean(united_features, dim=1, keepdim=True)
        max_out, _ = torch.max(united_features, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return x * attn, attn
    

class FrequencySplitFeatures(nn.Module):
    def __init__(self, channel: int, image_size: int):
        super(FrequencySplitFeatures, self).__init__()

        self.freq_slitter = LowHightFrequencyImageComponents((image_size, image_size))

        self.hlf = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, stride=1, padding=2, dilation=2, padding_mode='reflect'),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel // 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel // 2)
        )
        self.llf = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, stride=1, padding=2, dilation=2, padding_mode='reflect'),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel // 2, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel // 2)
        )

        self.up_feats = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        low_freq_features, hight_freq_features = self.freq_slitter(x)

        low_freq_features = self.llf(low_freq_features)
        hight_freq_features = self.hlf(hight_freq_features)

        united_features = low_freq_features + hight_freq_features
        united_features = self.up_feats(united_features)
        x = x + united_features
        x = nn.functional.leaky_relu(x)

        return x


class FFTChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FFTChannelAttention, self).__init__()

        self.preprocess = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=2, dilation=2, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, 3, 1, 1, padding_mode='reflect'),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Conv2d(channel, channel, 1, bias=False, dtype=torch.cfloat)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, bias=False),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(channel // reduction, channel, 1, bias=False)
        # )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False, dtype=torch.cfloat),
            RealImaginaryLeakyReLU(),
            nn.Linear(channel // reduction, channel, bias=False, dtype=torch.cfloat)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pre_fft = self.preprocess(x)

        z = torch.fft.fft2(pre_fft, norm='ortho')
        z = torch.fft.fftshift(z)

        z = self.conv1(z)
        z = real_imaginary_leakyrelu(z)

        # avg_out = self.fc(self.avg_pool(z).view(x.size(0), x.size(1)))
        # max_out = self.fc(self.max_pool(z).view(x.size(0), x.size(1)))

        _, max_indices = torch.nn.functional.adaptive_max_pool2d_with_indices(torch.abs(z), (1, 1))
        max_out = retrieve_elements_from_indices(z, max_indices)


        out = max_out
        out = self.fc(out.view(x.size(0),x.size(1)))
        attn = self.sigmoid(out.unsqueeze(2).unsqueeze(3))
        attn = torch.abs(attn)
        
        after_attn = pre_fft * attn
        out = after_attn + x

        return out
    

class FFTChannelAttentionV2(nn.Module):
    def __init__(self, channel: int, image_size: int, fsize: int = 8, reduction: int = 16):
        super(FFTChannelAttentionV2, self).__init__()

        pooling_depth = int(np.log2(image_size // fsize))
        
        self.pool_fft_features = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(channel, channel // 2, 3, padding=1, dtype=torch.cfloat),
                    RealImaginaryLeakyReLU(),
                    FFTMaxPool2D(2, 2),
                    nn.Conv2d(channel // 2, channel // 2 if i == pooling_depth - 1 else channel, 3, padding=1, dtype=torch.cfloat),
                    RealImaginaryLeakyReLU()
                )
                for i in range(pooling_depth)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(channel * fsize * fsize // 2, channel * fsize * fsize // 2 // reduction, dtype=torch.cfloat),
            RealImaginaryLeakyReLU(),
            nn.Linear(channel * fsize * fsize // 2 // reduction, channel, dtype=torch.cfloat),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = torch.fft.fft2(x, norm='forward')
        z = torch.fft.fftshift(z)

        z_deep_feats = self.pool_fft_features(z)
        z_deep_feats = z_deep_feats.view(x.size(0), -1)
        channel_attn = self.fc(z_deep_feats)
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = torch.abs(channel_attn.unsqueeze(2).unsqueeze(3))

        out = x * channel_attn

        with torch.no_grad():
            inv_attn = torch.abs(out - x).mean(dim=1).unsqueeze(1)
            inv_attn /= (inv_attn.max() + 1E-5)

        return x * channel_attn, inv_attn


class FFTCAFSModule(nn.Module):
    def __init__(self, image_size: int, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.fft_split_features = FrequencySplitFeatures(channel=channel, image_size=image_size)
        self.fft_ca = FFTChannelAttentionV2(channel=channel, reduction=reduction, image_size=image_size)

    def forward(self, x):
        x, ca_tensor = self.fft_ca(x)
        x = self.fft_split_features(x)
        return x, [ca_tensor]
