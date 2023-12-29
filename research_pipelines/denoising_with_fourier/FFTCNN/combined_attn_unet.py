from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn

from FFTCNN.attention import MixVitAttention as ComplexSpatialAttention, SpatialAttention, ChannelAttention, CBAM


padding_mode: str = 'reflect'



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    

def real_imaginary_relu(z):
    return nn.functional.relu(z.real) + 1.j * nn.functional.relu(z.imag)


class RealImaginaryReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return real_imaginary_relu(z) 


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


def gem(x, kernel_size: int, stride: int, p=3, eps=1e-6):
    return nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size, stride).pow(1.0 / p)


class GeneralizedMeanPooling2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int, p=3, eps=1e-6):
        super(GeneralizedMeanPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        x = gem(x, self.kernel_size, self.stride, p=self.p.clamp_min(1), eps=self.eps)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )
    

class Unet1lvl(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(Unet1lvl, self).__init__()

        self.process1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.process2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU()
        )

        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(mid_ch, mid_ch // 2, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(mid_ch // 2),
            nn.LeakyReLU()
        )

        self.process3 = nn.Sequential(
            nn.Conv2d(mid_ch // 2 + mid_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

        self.last_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.process1(x)
        yp1 = self.pool(y)
        ydf1 = self.process2(yp1)
        yup1 = self.up(ydf1)
        yup1 = torch.cat((yup1, y), dim=1)
        out = self.process3(yup1)
        out = self.last_conv(out)
        return out


def complex_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, dtype=torch.cfloat, padding_mode=padding_mode),
        RealImaginaryReLU(),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, dtype=torch.cfloat, padding_mode=padding_mode)
    )


class SpectralPooling(nn.Module):
    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.size(2), x.size(3)
        h = h_in // self.k
        w = w_in // self.k
        
        z = torch.fft.fft2(x, norm='ortho')
        
        z = torch.fft.fftshift(z)
        z = z[:, :, (h_in - h)//2:(h_in + h) // 2, (w_in - w)//2:(w_in + w)//2]
        z = torch.fft.ifftshift(z)

        new_x = torch.fft.ifft2(z, norm='ortho')
        new_x = new_x.real

        return new_x


class FFTAttention(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16, kernel_size: int = 7, window_size: int = 64, image_size: int = 256):
        super().__init__()
        self.fft_sa = ComplexSpatialAttention(in_ch=in_ch, patch_size=11, image_size=image_size)
        self.sa = SpatialAttention(kernel_size)
        self.final_ca = ChannelAttention(in_ch * 2, reduction)
        self.final_conv = nn.Conv2d(in_ch * 2, in_ch, 1)

    def forward(self, x):
        z = torch.fft.fft2(x, norm='ortho')
        z = torch.fft.fftshift(z)
        
        out_1, fft_attn = self.fft_sa(z)

        z = torch.fft.ifftshift(z)
        out_1 = torch.fft.ifft2(z, norm='ortho')
        out_1 = out_1.real

        out_2, float_sa = self.sa(x)

        out, _ = self.final_ca(torch.concat((out_1, out_2), dim=1))
        out = self.final_conv(out)

        with torch.no_grad():
            inv_attn = torch.abs(out_1 - x).mean(dim=1).unsqueeze(1)
            inv_attn /= inv_attn.max()

        return out, [fft_attn, torch.clamp(inv_attn, 0, 1), float_sa]


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()
        self.attn1 = FFTAttention(in_ch, window_size=window_size, image_size=image_size)
        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.norm1 = nn.BatchNorm2d(in_ch * 2)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(in_ch * 2, out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y, sa_1 = self.attn1(x)
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y, sa_1
    

class FeaturesProcessingWithLastConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, window_size=window_size, image_size=image_size)
        self.final_conv = conv1x1(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.features(x)
        y = self.final_conv(y)
        return y, sa


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, window_size=window_size, image_size=image_size)
        # self.pool = GeneralizedMeanPooling2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.features(x)
        y = self.pool(y)
        return y, sa
    

class FeaturesConvTransposeUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm = nn.BatchNorm2d(in_ch // 2)
        self.act = nn.LeakyReLU()
        self.features = FeaturesProcessing(in_ch // 2, out_ch, window_size=window_size, image_size=image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y = self.norm(y)
        y = self.act(y)
        y, sa = self.features(y)
        return y, sa


class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_size: int, image_size: int):
        super().__init__()
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.features = FeaturesProcessing(in_ch, out_ch, window_size=window_size, image_size=image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y, sa = self.features(y)
        return y, sa


class FFTAttentionUNetModule(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, need_up_features: bool = False, image_size: int = 256):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch, window_size=64, image_size=image_size)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch, window_size=64, image_size=image_size)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2, window_size=32, image_size=image_size // 2)
        self.downsample_block3 = FeaturesDownsample(mid_ch * 2, mid_ch * 3, window_size=16, image_size=image_size // 4)
        self.downsample_block4 = FeaturesDownsample(mid_ch * 3, mid_ch * 4, window_size=8, image_size=image_size // 8)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 4, mid_ch * 4, window_size=8, image_size=image_size // 8)

        self.upsample4 = FeaturesUpsample(mid_ch * 4, mid_ch * 3, window_size=16, image_size=image_size // 4)
        self.upsample3 = FeaturesUpsample(mid_ch * 3, mid_ch * 2, window_size=32, image_size=image_size // 2)
        self.upsample2 = FeaturesUpsample(mid_ch * 2, mid_ch, window_size=64 , image_size=image_size)
        self.upsample1 = FeaturesUpsample(mid_ch, mid_ch, window_size=64 , image_size=image_size)
        
        self.upsample_features_block4 = FeaturesProcessing(mid_ch * 3 + mid_ch * 3, mid_ch * 3, window_size=8, image_size=image_size // 8)
        self.upsample_features_block3 = FeaturesProcessing(mid_ch * 2 + mid_ch * 2, mid_ch * 2, window_size=16, image_size=image_size // 4)
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch, window_size=32, image_size=image_size // 2)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch, window_size=64 , image_size=image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx, sa_init = self.init_block(x)

        down_f1, sa_f1 = self.downsample_block1(hx)         # W // 2
        down_f2, sa_f2 = self.downsample_block2(down_f1)    # W // 4
        down_f3, sa_f3 = self.downsample_block3(down_f2)    # W // 8
        down_f4, sa_f4 = self.downsample_block4(down_f3)    # W // 16

        deep_f, sa_f = self.deep_conv_block(down_f4)

        deep_f, sa_up_4 = self.upsample4(deep_f)
        decoded_f4 = torch.cat((down_f3, deep_f), axis=1)
        decoded_f4, sa_df4 = self.upsample_features_block4(decoded_f4)

        deep_f, sa_up_3 = self.upsample3(decoded_f4)
        decoded_f3 = torch.cat((down_f2, deep_f), axis=1)
        decoded_f3, sa_df3 = self.upsample_features_block3(decoded_f3)

        deep_f, sa_up_2 = self.upsample2(decoded_f3)
        decoded_f2 = torch.cat((down_f1, deep_f), axis=1)
        decoded_f2, sa_df2 = self.upsample_features_block2(decoded_f2)

        deep_f, sa_up_1 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, deep_f), dim=1)
        decoded_f1, sa_df1 = self.upsample_features_block1(decoded_f1)

        return decoded_f1, sa_init + sa_f1 + sa_f2 + sa_f3 + sa_f4 + sa_f + sa_up_4 + sa_up_3 + sa_up_2 + sa_up_1 + sa_df4 + sa_df3 + sa_df2 + sa_df1


class FFTAttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3,  out_ch: int = 3, image_size: int = 256):
        super().__init__()

        middle_channels = 16

        self.in_conv = nn.Conv2d(in_ch, middle_channels, 1)
        self.unet = FFTAttentionUNetModule(middle_channels, middle_channels * 2, middle_channels, image_size=image_size)
        self.out_conv = nn.Conv2d(middle_channels, out_ch, 1)
        self.export = False

    def to_export(self):
        self.export = True

    def norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 - 1

    def denorm_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1) * 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.norm_input(x)

        y = self.in_conv(hx)
        y, sa_list = self.unet(y)
        y = self.out_conv(y)

        if self.export:
            return self.denorm_input(hx + y)

        if self.training:
            sa_list = [
                nn.functional.interpolate(torch.abs(sa), (x.size(2), x.size(3)))
                for sa in sa_list
            ]

        return self.denorm_input(hx + y), sa_list



if __name__ == '__main__':
    import cv2
    import numpy as np
    from timeit import default_timer as time
    import segmentation_models_pytorch as smp


    model = FFTAttentionUNet(3, 3)

    m_params = sum(p.numel() for p in model.parameters())

    print('Ours model params: {}M'.format(m_params // 10 ** 6))

    unet_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=3,
    )
    unet_model_params = sum(p.numel() for p in unet_model.parameters())

    print('UNet(ResNet18) model params: {}M'.format(unet_model_params // 10 ** 6))

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

    with torch.no_grad():
        out = model(t)

    start_time = time()
    with torch.no_grad():
        out = model(t)
    finish_time = time()

    infer_time = finish_time - start_time
    print('Inference time: {:.5f} sec'.format(infer_time))

    model.to_export()
    model.eval()

    traced = torch.jit.trace(model, example_inputs=t)
    torch.jit.save(traced, '/home/alexey/Downloads/fftcnn.pt')

    gpool = GeneralizedMeanPooling2d(2, 2)
    print(gpool(t).shape)
