import torch
from torch import nn
import numpy

from .spatial_cnn import SpatialMiniUNet


padding_mode = 'reflect'


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def custom_softmax(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(x))


def custom_swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(x)


def real_imaginary_relu(z):
    return nn.functional.relu(z.real) + 1.j * nn.functional.relu(z.imag)


def phase_amplitude_relu(z):
    return nn.functional.relu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))


def real_imaginary_swish(z):
    return nn.functional.silu(z.real) + 1.j * nn.functional.silu(z.imag)


def phase_amplitude_swish(z):
    return nn.functional.silu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))


class InstanceNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels, dtype=torch.cfloat))
            self.shift = nn.Parameter(torch.zeros(channels, dtype=torch.cfloat))

    
    def forward(self, x):
        batch_size = x.size(0)
        xh = x.size(2)
        xw = x.size(3)
        assert self.channels == x.size(1)
        x = x.view(batch_size, self.channels, -1)
        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.channels, -1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(batch_size, self.channels, xh, xw)


class FFTDownsample(nn.Module):
    def __init__(self, k: int = 2):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_in, w_in = x.size(2), x.size(3)
        h = h_in // self.k
        w = w_in // self.k
        
        new_y = torch.zeros(x.size(0), x.size(1), h, w, dtype=x.dtype, device=x.device)
        
        y = torch.fft.fftshift(x)
        new_y = y[:, :, (h_in - h)//2:(h_in + h) // 2, (w_in - w)//2:(w_in + w)//2]
        new_y = torch.fft.ifftshift(new_y)

        return new_y
    
    
class FFTUpsample(nn.Module):
    def __init__(self, k: int = 2) -> None:
        super().__init__()
        self.k = k

    def forward(self, x):
        h_in, w_in = x.size(2), x.size(3)
        h = h_in * self.k
        w = w_in * self.k
        
        new_y = torch.zeros(x.size(0), x.size(1), h, w, dtype=x.dtype, device=x.device)
        
        y = torch.fft.fftshift(x)
        new_y[:, :, (h - h_in)//2:(h + h_in) // 2, (w - w_in)//2:(w + w_in)//2] = y
        new_y = torch.fft.ifftshift(new_y)
        
        return new_y


class FFTConv(nn.Module):
    def __init__(self, inplanes: int, planes: int, H: int, W: int, bias: bool = True) -> None:
        super().__init__()
        self.planes = planes
        self.filter = nn.Parameter(torch.zeros(inplanes, planes, H, W, dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(inplanes, planes, H, W, dtype=torch.cfloat))
        else:
            self.bias = None

    def forward(self, z):
        bs = z.size(0)
        ch = z.size(1)
        h, w = z.size(2), z.size(3)

        result = torch.zeros(bs, self.planes, h, w, dtype=torch.cfloat, device=z.device)

        for plane in range(self.planes):
            plane_value = torch.stack(
                [
                    z[:, inplane] * self.filter[inplane, plane].unsqueeze(0).repeat(bs, 1, 1) +\
                        (0 if self.bias is None else self.bias[inplane, plane].unsqueeze(0).repeat(bs, 1, 1))
                    for inplane in range(ch)
                ],
                dim=0
            ).sum(dim=0)
            result[:, plane] = plane_value
        
        return result


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, H: int, W: int):
        super().__init__()
        self.conv1 = FFTConv(in_ch, in_ch*2, H, W)
        self.norm1 = InstanceNorm(in_ch * 2)
        self.act1 = real_imaginary_swish
        self.conv2 = FFTConv(in_ch * 2, out_ch, H, W)
        self.norm2 = InstanceNorm(out_ch)

        self.down_bneck = FFTConv(in_ch, out_ch, H, W)

        self.act_final = real_imaginary_swish

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y


class FeaturesProcessingWithLastConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, H: int, W: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, H, W)
        self.final_conv = nn.Conv2d(out_ch, out_ch, 1, dtype=torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.final_conv(y)
        return y


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, H: int, W: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch, H, W)
        self.downsample = FFTDownsample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.downsample(y)
        return y
    

class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, H: int, W: int):
        super().__init__()
        self.upsample = FFTUpsample()
        self.features = FeaturesProcessing(in_ch, out_ch, H * 2, W * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.upsample(x)
        y = self.features(y)
        y = real_imaginary_swish(y)
        return y


class MiniUNet(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, H: int, W: int):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch, H, W)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch, H, W)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2, H // 2, W // 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch, H // 4, W // 4)

        self.upsample2 = FeaturesUpsample(mid_ch, mid_ch, H // 4, W // 4)
        self.upsample1 = FeaturesUpsample(mid_ch, mid_ch, H // 2, W // 2)
        
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch, H // 2, W // 2)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch, H, W)

        self.final_conv = nn.Conv2d(out_ch, out_ch, 1, dtype=torch.cfloat)

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

        return decoded_f1, []

    
class FourierBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        H: int, W: int
    ) -> None:
        super().__init__()
        self.upsample_conv = nn.Conv2d(inplanes, planes, 1, dtype=torch.cfloat)
        self.process = MiniUNet(planes, planes // 2, planes, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.upsample_conv(x)

        out, _ = self.process(identity)

        out += identity
        out = real_imaginary_swish(out)

        return out


class FFTCNN(nn.Module):
    def __init__(self, H: int, W: int, four_normalized=True):
        super().__init__()

        self.four_normalized = 'ortho' if four_normalized else None        

        self.cycb_unet = SpatialMiniUNet(2, 16, 2)
        
        # self.basic_layer_1 = FourierBlock(3, 16, H, W)
        self.basic_layer_1 = FeaturesProcessing(1, 16, H, W)
        self.final_comples_conv = nn.Conv2d(16, 1, 1, dtype=torch.cfloat)
        self.final_real_conv = nn.Conv2d(3, 3, 1, dtype=torch.float32)
            
    def get_fourier(self, x):
        fourier_transform_x = torch.fft.fft2(
            x, norm=self.four_normalized
        )
        return fourier_transform_x

    def forward(self, image):
        y_inp = image[:, :1]
        crcb_inp = image[:, 1:]

        inp = self.get_fourier(y_inp)
        
        x = self.basic_layer_1(inp)
        x = self.final_comples_conv(x)

        x = inp / x

        restored_x = torch.fft.ifft2(
            x, norm=self.four_normalized
        )

        crcb_x = crcb_inp - self.cycb_unet(crcb_inp)

        restored_x = torch.concat((torch.abs(restored_x), crcb_x), dim=1)

        restored_x = self.final_real_conv(restored_x)

        return restored_x, x


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.onnx import OperatorExportTypes

    device = 'cpu'

    wsize = 512

    model = FFTCNN(wsize, wsize).to(device)
    model.apply(init_weights)

    img_path = '/home/alexey/Downloads/image1 (1).jpg'
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)
    inp = inp.repeat(4, 1, 1, 1)

    conv = FFTConv(3, 5, wsize, wsize).to(device)

    with torch.no_grad():
        out = conv(inp)

    print(out.shape)

    with torch.no_grad():
        out, _ = model(inp)


    print('Rand L1: {}'.format(torch.nn.functional.l1_loss(out, inp).item()))
