from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn

from FFTCNN.attention import CBAM


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


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.attn1 = CBAM(in_ch, reduction=8)
        
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
            y, _, sa_1 = self.attn1(x)
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
        return y, [sa_1]
    

class FeaturesProcessingWithLastConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, use_attention=use_attention)
        self.final_conv = conv1x1(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.features(x)
        y = self.final_conv(y)
        return y, sa


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, use_attention=use_attention)
        self.pool = GeneralizedMeanPooling2d(2, 2)
        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, sa = self.features(x)
        y = self.pool(y)
        return y, sa
    

class FeaturesConvTransposeUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(in_ch // 2)
        self.act = nn.LeakyReLU()
        self.features = FeaturesProcessing(in_ch // 2, out_ch, use_attention=use_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y = self.norm(y)
        y = self.act(y)
        y, sa = self.features(y)
        return y, sa


class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.features = FeaturesProcessing(in_ch, out_ch, use_attention=use_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y, sa = self.features(y)
        return y, sa


class AttentionUNetModule(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, need_up_features: bool = False, use_attention: bool = True):
        super().__init__()

        self.init_block = FeaturesProcessing(in_ch, mid_ch, use_attention=False)
        self.init_block_with_attn = FeaturesProcessing(mid_ch, mid_ch, use_attention=use_attention)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch, use_attention=use_attention)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2, use_attention=use_attention)
        self.downsample_block3 = FeaturesDownsample(mid_ch * 2, mid_ch * 3, use_attention=use_attention)
        self.downsample_block4 = FeaturesDownsample(mid_ch * 3, mid_ch * 4, use_attention=use_attention)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 4, mid_ch * 4, use_attention=use_attention)

        self.upsample4 = FeaturesUpsample(mid_ch * 4, mid_ch * 3, use_attention=use_attention)
        self.upsample3 = FeaturesUpsample(mid_ch * 3, mid_ch * 2, use_attention=use_attention)
        self.upsample2 = FeaturesUpsample(mid_ch * 2, mid_ch, use_attention=use_attention)
        self.upsample1 = FeaturesUpsample(mid_ch, mid_ch, use_attention=use_attention)
        
        self.upsample_features_block4 = FeaturesProcessing(mid_ch * 3 + mid_ch * 3, mid_ch * 3, use_attention=use_attention)
        self.upsample_features_block3 = FeaturesProcessing(mid_ch * 2 + mid_ch * 2, mid_ch * 2, use_attention=use_attention)
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch, use_attention=use_attention)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch, use_attention=use_attention)

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


class AttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3,  out_ch: int = 3, use_attention: bool = True):
        super().__init__()

        self.unet = AttentionUNetModule(in_ch, 16, out_ch, use_attention=use_attention)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1)
        self.export = False
        self.use_attention = use_attention

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
            return self.denorm_input(hx + y)

        if self.training and self.use_attention:
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


    model = AttentionUNet(3, 3, use_attention=False)

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

    # sa_tensors = []
    # for k in range(len(sa_list) // 4):
    #     sa_tensors.append(torch.concat([sa_list[2*k + q] for q in range(4)], dim=3))

    # sa_tensor = torch.concat(sa_tensors, dim=2)
    # print(sa_tensor.shape)

    print(out[0].shape)

    _ = model.eval()

    with torch.no_grad():
        out = model(t)

    n_attempts: int = 10

    start_time = time()
    with torch.no_grad():
        for _ in range(n_attempts):
            out = model(t)
    finish_time = time()

    infer_time = (finish_time - start_time) / n_attempts
    print('Inference time: {:.5f} sec'.format(infer_time))

    model.to_export()
    model.eval()

    traced = torch.jit.trace(model, example_inputs=t)
    torch.jit.save(traced, '/home/alexey/Downloads/fftcnn.pt')

    gpool = GeneralizedMeanPooling2d(2, 2)
    print(gpool(t).shape)
