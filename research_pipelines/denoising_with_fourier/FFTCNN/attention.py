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

from typing import Tuple, Union, List

import numpy as np
import math
import torch
from torch import nn
import scipy.linalg

from utils.haar_utils import HaarForward, HaarInverse


def sim_attention (X: torch.Tensor, lamb: float) -> Tuple[torch.Tensor, torch.Tensor]:
    n = X.shape[2] * X.shape[3] - 1
    d = (X - X.mean(dim=[2,3]).unsqueeze(2).unsqueeze(3)).pow(2)
    v = d.sum(dim=[2,3]).unsqueeze(2).unsqueeze(3) / n
    E_inv = d / (4 * (v + lamb)) + 0.5
    out = X * torch.sigmoid(E_inv)

    with torch.no_grad():
        att = (X - out).mean(dim=1).unsqueeze(dim=1)
        att = att / (att.max() + 1E-5)

    return out, att


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


def retrieve_complex_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    real_output = flattened_tensor.real.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    imag_output = flattened_tensor.imag.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    output = real_output + 1.0j * imag_output
    return output


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                    y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel


def real_imaginary_leakyrelu(z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return nn.functional.leaky_relu(z_real), nn.functional.leaky_relu(z_imag)


def rfftshift(z: torch.Tensor) -> torch.Tensor:
    z[:, :, :z.size(2) // 2] = torch.flip(z[:, :, :z.size(2) // 2], dims=(2,))
    z[:, :, z.size(2) // 2:] = torch.flip(z[:, :, z.size(2) // 2:], dims=(2,))
    return z


def get_even_index(n: int) -> int:
    if n % 2 == 0:
        return n // 2 + 1
    return n // 2


class MatrixRFFT(nn.Module):
    def __init__(self, N: int):
        super().__init__()
        assert N > 0, 'Size of signal must be not 0'

        W = torch.from_numpy(scipy.linalg.dft(N)).to(torch.cfloat)
        Wr, Wi = W.real.unsqueeze(0), W.imag.unsqueeze(0)

        self.real_w = torch.nn.Parameter(Wr, requires_grad=False)
        self.imag_w = torch.nn.Parameter(Wi, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fft_size = get_even_index(x.size(3))
        # fft_size = x.size(3) // 2 + 1

        z_real = (
            (self.real_w @ x) @ self.real_w.transpose(1, 2)[:, :, :fft_size] - 
            (self.imag_w @ x) @ self.imag_w.transpose(1, 2)[:, :, :fft_size]
        ) / x.size(2) / x.size(3)

        z_imag = (
            (self.imag_w @ x) @ self.real_w.transpose(1, 2)[:, :, :fft_size] + 
            (self.real_w @ x) @ self.imag_w.transpose(1, 2)[:, :, :fft_size]
        ) / x.size(2) / x.size(3)

        return z_real, z_imag


class RealImaginaryLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_real, z_imag = z
        return real_imaginary_leakyrelu(z_real, z_imag)
    

class ComplexConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, bias: bool = True, stride: int = 1, padding: int = 0, padding_mode: str = 'zeros'):
        super().__init__()

        self.real_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, 
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode=padding_mode,
            bias=False
        )
        self.imag_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, 
            kernel_size=kernel_size, stride=stride,
            padding=padding, padding_mode=padding_mode,
            bias=False
        )

        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_ch, 1, 1))
            self.imag_bias = nn.Parameter(torch.zeros(out_ch, 1, 1))
        else:
            self.real_bias = None
            self.imag_bias = None

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
         z_real, z_imag = z

         out_real = self.real_conv(z_real) - self.imag_conv(z_imag)
         out_imag = self.imag_conv(z_real) + self.real_conv(z_imag)

         if self.imag_bias is not None:
             out_real = out_real + self.real_bias
             out_imag = out_imag + self.imag_bias

         return out_real, out_imag
    

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.real_fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.imag_fc = nn.Linear(in_features=in_features, out_features=out_features, bias=False)

        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_features, 1))
            self.imag_bias = nn.Parameter(torch.zeros(out_features, 1))
        else:
            self.real_bias = None
            self.imag_bias = None

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_real, z_imag = z

        out_real = self.real_fc(z_real) - self.imag_fc(z_imag)
        out_imag = self.imag_fc(z_real) + self.real_fc(z_imag)

        if self.imag_bias is not None:
             out_real = out_real + self.real_bias
             out_imag = out_imag + self.imag_bias

        return out_real, out_imag
    

class FFTMaxPool2D(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, z: Tuple[torch.Tensor, torch.Tensor], return_indices: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        z_real, z_imag = z
        z_abs = z_real * z_real + z_imag * z_imag

        _, max_indices = torch.nn.functional.max_pool2d_with_indices(z_abs, self.kernel_size, self.stride)

        max_real_out = retrieve_elements_from_indices(z_real, max_indices)
        max_imag_out = retrieve_elements_from_indices(z_imag, max_indices)

        max_out = (max_real_out, max_imag_out)

        if return_indices:
            return max_out, max_indices

        return max_out


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

        out = x * attn

        with torch.no_grad():
            vis_att = torch.abs(out - x).mean(dim=1).unsqueeze(dim=1)
            vis_att = vis_att / (vis_att.max() + 1E-5)

        return out, vis_att


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
    

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B x C x(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B x C x (W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B x (N) x (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B x C x N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        pre_out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*pre_out + x

        with torch.no_grad():
            inv_attn = torch.abs(x - out).mean(dim=1).unsqueeze(1)
            inv_attn /= (inv_attn.max() + 1E-5)

        return out, inv_attn
    

class WaveletSpaialAttentionV2(nn.Module):
    padding_mode = 'reflect'

    def get_ksize(self, image_size: int) -> int:
        if image_size >= 128:
            return 7
        elif image_size >= 32:
            return 5
        return 3

    def __init__(self, channel: int, image_size: int):
        super(WaveletSpaialAttentionV2, self).__init__()

        self.wavelet_forward = HaarForward()
        self.wavelet_inverse = HaarInverse()

        self.features_to_sa = nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.ll_feats = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.lh_feats = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.hl_feats = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
        self.hh_feats = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

        self.ll_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.lh_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.hl_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.hh_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))

        self.conv_last = nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)

    def forward(self, x: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        w_feats = self.wavelet_forward(x)

        y = self.features_to_sa(w_feats)

        ll_y = self.ll_feats(y)
        lh_y = self.lh_feats(y)
        hl_y = self.hl_feats(y)
        hh_y = self.hh_feats(y)

        w_ll, ll_attn = self.ll_sa(ll_y)
        w_lh, lh_attn = self.lh_sa(lh_y)
        w_hl, hl_attn = self.hl_sa(hl_y)
        w_hh, hh_attn = self.hh_sa(hh_y)

        # ll_y = w_feats[:, :x.size(1)]               * ll_attn
        # lh_y = w_feats[:, x.size(1):x.size(1)*2]    * lh_attn
        # hl_y = w_feats[:, x.size(1)*2:x.size(1)*3]  * hl_attn
        # hh_y = w_feats[:, x.size(1)*3:]             * hh_attn

        # y = torch.cat([ll_y, lh_y, hl_y, hh_y], dim=1)
        y = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=1)

        y = w_feats + self.conv_last(y)

        y = self.wavelet_inverse(y)

        with torch.no_grad():
            attn = torch.cat(
                [
                    torch.cat([ll_attn, lh_attn], dim=3),
                    torch.cat([hl_attn, hh_attn], dim=3)
                ],
                dim=2
            )

        return y, attn    


class WaveletSpaialAttentionV2Light(nn.Module):
    padding_mode = 'reflect'

    def get_ksize(self, image_size: int) -> int:
        if image_size >= 128:
            return 7
        elif image_size >= 32:
            return 5
        return 3

    def __init__(self, channel: int, image_size: int):
        super(WaveletSpaialAttentionV2Light, self).__init__()

        self.wavelet_forward = HaarForward()
        self.wavelet_inverse = HaarInverse()

        self.features_to_sa = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU(),
            nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode, groups=4),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU()
        )

        self.ll_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.lh_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.hl_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))
        self.hh_sa = SpatialAttention(kernel_size=self.get_ksize(image_size))

        self.conv_last = nn.Conv2d(channel * 4, channel * 4, kernel_size=1, stride=1, groups=4)

    def forward(self, x: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        w_feats = self.wavelet_forward(x)

        y = self.features_to_sa(w_feats)

        ll_y = y[:, :x.size(1)]
        lh_y = y[:, x.size(1):x.size(1)*2]
        hl_y = y[:, x.size(1)*2:x.size(1)*3]
        hh_y = y[:, x.size(1)*3:]

        _, ll_attn = self.ll_sa(ll_y)
        _, lh_attn = self.lh_sa(lh_y)
        _, hl_attn = self.hl_sa(hl_y)
        _, hh_attn = self.hh_sa(hh_y)

        w_ll = w_feats[:, :x.size(1)]               * ll_attn
        w_lh = w_feats[:, x.size(1):x.size(1)*2]    * lh_attn
        w_hl = w_feats[:, x.size(1)*2:x.size(1)*3]  * hl_attn
        w_hh = w_feats[:, x.size(1)*3:]             * hh_attn

        y = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=1)

        y = self.conv_last(y)

        y = self.wavelet_inverse(y)

        with torch.no_grad():
            attn = torch.cat(
                [
                    torch.cat([ll_attn, lh_attn], dim=3),
                    torch.cat([hl_attn, hh_attn], dim=3)
                ],
                dim=2
            )

        return y, attn
    

class Unet1lvl(nn.Module):
    padding_mode = 'reflect'
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, activation: nn.Module = nn.Identity()):
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
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(mid_ch // 2),
            nn.LeakyReLU()
        )

        self.process3 = nn.Sequential(
            nn.Conv2d(mid_ch // 2 + mid_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(out_ch)
        )
        self.last_act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.process1(x)
        yp1 = self.pool(y)
        ydf1 = self.process2(yp1)
        yup1 = self.up(torch.nn.functional.interpolate(ydf1, scale_factor=2, mode='bilinear'))
        yup1 = torch.cat((yup1, y), dim=1)
        out = self.process3(yup1)
        out = self.last_act(out)
        return out


class WaveletSpaialAttentionV4(nn.Module):
    padding_mode = 'reflect'

    def get_ksize(self, image_size: int) -> int:
        if image_size >= 128:
            return 7
        elif image_size >= 32:
            return 5
        return 3

    def __init__(self, channel: int, image_size: int):
        super(WaveletSpaialAttentionV4, self).__init__()

        self.wavelet_forward = HaarForward()
        self.wavelet_inverse = HaarInverse()

        self.in_feats = nn.Sequential(
            nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU(),
            nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.BatchNorm2d(channel * 4),
            nn.LeakyReLU(),
        )

        self.features_to_sa = Unet1lvl(channel * 4, channel, 4, activation=torch.sigmoid)

        self.conv_last = nn.Conv2d(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode)


    def forward(self, x: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        w_feats = self.wavelet_forward(x)

        y = self.in_feats(w_feats)

        attn_maps = self.features_to_sa(y)

        ll_attn = attn_maps[:, 0].unsqueeze(1)
        lh_attn = attn_maps[:, 1].unsqueeze(1)
        hl_attn = attn_maps[:, 2].unsqueeze(1)
        hh_attn = attn_maps[:, 3].unsqueeze(1)

        ll_y = y[:, :x.size(1)]               * ll_attn
        lh_y = y[:, x.size(1):x.size(1)*2]    * lh_attn
        hl_y = y[:, x.size(1)*2:x.size(1)*3]  * hl_attn
        hh_y = y[:, x.size(1)*3:]             * hh_attn

        y = torch.cat([ll_y, lh_y, hl_y, hh_y], dim=1)

        y = w_feats + self.conv_last(w_feats)

        y = self.wavelet_inverse(y)

        with torch.no_grad():
            attn = torch.cat(
                [
                    torch.cat([ll_attn, lh_attn], dim=3),
                    torch.cat([hl_attn, hh_attn], dim=3)
                ],
                dim=2
            )

        return y, attn


class ShuffledSelfAttention(nn.Module):
    padding_mode = 'reflect'

    def __init__(self, channel: int, image_size: int):
        super(ShuffledSelfAttention, self).__init__()

        scale_factor = 4 if image_size >= 64 else 2
        self.pix_unshuffle = nn.PixelUnshuffle(downscale_factor=scale_factor)
        self.pix_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)

        self.self_attn = Self_Attn(in_dim=channel * scale_factor ** 2)

    def forward(self, x: torch.Tensor) ->  Tuple[torch.Tensor, torch.Tensor]:
        y_pathes = self.pix_unshuffle(x)
        y_pathes, _ = self.self_attn(y_pathes)
        y = self.pix_shuffle(y_pathes)

        with torch.no_grad():
            inv_attn = torch.abs(x - y).mean(dim=1).unsqueeze(1)
            inv_attn /= (inv_attn.max() + 1E-5)

        return y, inv_attn


class ForFFTPad(nn.Module):
    def __init__(self, padding: int):
        super(ForFFTPad, self).__init__()
        assert padding > 0
        self.padding = padding

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_real, z_imag = z

        z_real = torch.cat([torch.flip(z_real[:, :, :, :self.padding], dims=(2, 3)), z_real], dim=3)
        z_real = torch.nn.functional.pad(z_real, (0, self.padding, self.padding, self.padding), mode='constant', value=0)

        z_imag = torch.cat([torch.flip(z_imag[:, :, :, :self.padding], dims=(2, 3)), z_imag], dim=3)
        z_imag = torch.nn.functional.pad(z_imag, (0, self.padding, self.padding, self.padding), mode='constant', value=0)

        return z_real, z_imag
    

class ResidualComplexConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.pad = ForFFTPad(1)
        self.conv = ComplexConv(in_ch, out_ch, 3, padding=0)
        self.bn_re = nn.BatchNorm2d(out_ch)
        self.bn_im = nn.BatchNorm2d(out_ch)

        self.bottleneck = ComplexConv(in_ch, out_ch, 1, padding=0, bias=False)


    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.pad(z)
        y = self.conv(y)

        y = (self.bn_re(y[0]), self.bn_im(y[1]))

        y_b = self.bottleneck(z)
        y = (y[0] + y_b[0], y[1] + y_b[1])

        return y


class RealFFTChannelAttentionV4(nn.Module):
    def __init__(self, channel: int, image_size: int, fsize: int = 8, reduction: int = 16):
        super(RealFFTChannelAttentionV4, self).__init__()

        self.real_fft = MatrixRFFT(N=image_size)

        pooling_depth = int(np.log2(image_size // fsize))
        
        self.pool_fft_features = nn.Sequential(
            *[
                nn.Sequential(
                    ResidualComplexConv(channel, channel // 2),
                    FFTMaxPool2D(2, 2),
                    ResidualComplexConv(channel // 2, channel // 2 if i == pooling_depth - 1 else channel),
                )
                for i in range(pooling_depth)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(channel * fsize * fsize // 2 // 2, channel * fsize * fsize // 2 // 2 // reduction),
            nn.LeakyReLU(),
            nn.Linear(channel * fsize * fsize // 2 // 2 // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.real_fft(x)

        z_deep_feats = self.pool_fft_features(z)
        z_deep_feats = (z_deep_feats[0].view(x.size(0), -1), z_deep_feats[1].view(x.size(0), -1))

        z_abs_feats = torch.sqrt(z_deep_feats[0] * z_deep_feats[0] + z_deep_feats[1] * z_deep_feats[1])

        channel_attn = self.fc(z_abs_feats)
        channel_attn = self.sigmoid(channel_attn)
        channel_attn = channel_attn.unsqueeze(2).unsqueeze(3)

        out = x * channel_attn

        with torch.no_grad():
            inv_attn = torch.abs(out - x).mean(dim=1).unsqueeze(1)
            inv_attn /= (inv_attn.max() + 1E-5)

        return x * channel_attn, inv_attn
    

class FCABlock(nn.Module):
    def __init__(self, channel: int, image_size: int):
        super().__init__()

        self.real_fft = MatrixRFFT(N=image_size)

        self.in_feats = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(channel, channel, 1)
        self.fc = nn.Linear(channel, channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_hf_feats = self.in_feats(x)

        complex_complex_hf_feats = self.real_fft(init_hf_feats)
        hf_spectrums = torch.sqrt(torch.pow(complex_complex_hf_feats[0], 2) + torch.pow(complex_complex_hf_feats[1], 2))
        hf_spectrums = nn.functional.relu(self.conv(hf_spectrums))
        hf_feats = self.pool(hf_spectrums).view(x.size(0), x.size(1))
        channels_probs = nn.functional.sigmoid(self.fc(hf_feats)).unsqueeze(2).unsqueeze(3)

        weighted_channels = init_hf_feats * channels_probs

        out = x + weighted_channels

        with torch.no_grad():
            attn = weighted_channels.mean(dim=1).unsqueeze(1)
            attn = attn / (attn.max() + 1e-5)
        return out, attn


class FFTCAFSModule(nn.Module):
    def __init__(self, image_size: int, channel: int, reduction: int = 16, kernel_size: int = 7, mode: str = 'full') -> None:
        super().__init__()
        assert mode in ['full', 'ca', 'sa', 'cbam', 'fca', 'none']

        if mode in ['full', 'ca', 'sa']:
            self.in_feats = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU()
            )
            self.final_conv = nn.Conv2d(channel, channel, 1, stride=1, padding=0)

        if mode in ['full', 'ca']:
            self.fft_ca = RealFFTChannelAttentionV4(channel=channel, reduction=reduction, image_size=image_size)

        if mode in ['full', 'sa']:
            self.fft_sa = WaveletSpaialAttentionV2Light(channel=channel, image_size=image_size)

        if mode == 'cbam':
            self.cbam = CBAM(channel=channel, reduction=reduction, kernel_size=kernel_size)

        if mode == 'fca':
            self.fca_block = FCABlock(channel=channel, image_size=image_size)

        self.mode = mode
        self.forward_methods = {
            'full':     self.own_full_forward,
            'ca':       self.own_ca_forward,
            'sa':       self.own_sa_forward,
            'cbam':     self.cbam_unet_forward,
            'fca':      self.fca_unet_forward,
            'none':     self.classic_unet_forward
        }

        self.forward = self.forward_methods[self.mode]

    def classic_unet_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Original U-Net (without attention)
        with torch.no_grad():
            att = x.mean(dim=1).unsqueeze(dim=1)
            att = att / (att.max() + 1E-5)
        return x, [att]
        
    def cbam_unet_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # U-Net with CBAM
        y, ca_tensor, sa_tensor = self.cbam(x)
        return y, [ca_tensor, sa_tensor]
    
    def fca_unet_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Use only time-frequency SA
        y, attn_tensor = self.fca_block(x)
        return y, [attn_tensor]
        
    def own_ca_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Use only frequency CA
        y, ca_tensor = self.fft_ca(x)
        return y, [ca_tensor]
    
    def own_sa_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Use only time-frequency SA
        y, sa_tensor = self.fft_sa(x)
        return y, [sa_tensor]

    def own_full_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Use proposed attention module
        y = self.in_feats(x)
        y, ca_tensor = self.fft_ca(y)
        y, sa_tensor = self.fft_sa(y)
        y = x + self.final_conv(y)
        return y, [ca_tensor, sa_tensor]


if __name__ == '__main__':
    import cv2
    from timeit import default_timer as time
    import scipy.linalg

    layer = FFTCAFSModule(256, 32, mode='fca')
    out, attn = layer(torch.rand(1, 32, 256, 256))

    exit(0)

    # torch.set_printoptions(precision=4, sci_mode=False)

    def DFT_matrices(N):
        i, j = torch.meshgrid(torch.arange(N), torch.arange(N))
        omega = torch.ones(N, N) * torch.FloatTensor([- 2 * torch.pi / N])
        return torch.cos( omega * i * j ).to(torch.float32), torch.sin( omega * i * j ).to(torch.float32)

    image_path = '/media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/DIV2K_0134.png'
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    N = 256
    # x = torch.rand(2, 3, N, N, dtype=torch.float32) * 256
    x = rgb[:, :, :N, :N]
    W = torch.from_numpy(scipy.linalg.dft(N)).to(torch.cfloat)
    # Wr, Wi = DFT_matrices(N)
    Wr = W.real.unsqueeze(0)
    Wi = W.imag.unsqueeze(0)


    # r = torch.stack(
    #     [
    #         torch.bmm(torch.bmm(Wr, x[:, q]), Wr.transpose(1, 2)) - 
    #         torch.bmm(torch.bmm(Wi, x[:, q]), Wi.transpose(1, 2)) 
    #         for q in range(x.size(1))
    #     ],
    #     dim=1
    # )
    r = ((Wr @ x) @ Wr.transpose(1, 2)[:, :, :get_even_index(N)] - (Wi @ x) @ Wi.transpose(1, 2)[:, :, :get_even_index(N)]) / N / N
    i = ((Wi @ x) @ Wr.transpose(1, 2)[:, :, :get_even_index(N)] + (Wr @ x) @ Wi.transpose(1, 2)[:, :, :get_even_index(N)]) / N / N

    print(r.shape)

    mf = MatrixRFFT(N)

    f = torch.fft.rfft2(x.to('cuda'), norm='forward').to('cpu')
    f2 = mf(x)

    reps = 0.001
    aeps = 1e-3
    print(torch.allclose(f.real, r, rtol=reps, atol=aeps))
    print(torch.allclose(f.real, f2[0], rtol=reps, atol=aeps))

    print()

    print(torch.allclose(f.imag, i, rtol=reps, atol=aeps))
    print(torch.allclose(f.imag, f2[1], rtol=reps, atol=aeps))

    print(torch.abs(f.real - r).max())
    print(torch.abs(f.imag - i).max())