import torch
from torch import nn
import numpy

from FFTCNN.cbam import CBAM

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


class FourierComplexConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ksize: int, paddings: int = 0, use_bias: bool = True):
        super().__init__()
        
        self.conv_w = nn.Parameter(torch.randn(out_ch, in_ch, ksize, ksize).to(torch.cfloat))
        if use_bias:
            self.conv_b = nn.Parameter(torch.randn(out_ch).to(torch.cfloat))
        else:
            self.conv_b = None
        
        self.paddings = paddings
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x,
            self.conv_w,
            self.conv_b,
            padding=self.paddings
        )
    
    
# def conv1x1(inplanes, planes):
#     return FourierComplexConv2d(inplanes, planes, 1, 0)
    
    
# def conv3x3(inplanes, planes):
#     return FourierComplexConv2d(inplanes, planes, 3, 1)



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
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.act1 = nn.Mish()
        self.conv2 = conv3x3(in_ch * 2, out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)

        hx = self.down_bneck(hx)

        y = self.act_final(hx + y)
        return y
    

class FeaturesProcessingWithLastConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch)
        self.final_conv = conv1x1(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.final_conv(y)
        return y


class FeaturesDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.features = FeaturesProcessing(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        y = self.pool(y)
        return y
    

class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.act = nn.Mish()
        self.features = FeaturesProcessing(in_ch // 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y = self.act(y)
        y = self.features(y)
        return y


class MiniUNet(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2) # FeaturesUpsample(mid_ch, mid_ch)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2) # FeaturesUpsample(mid_ch, mid_ch)

        self.attn2 = CBAM(mid_ch + mid_ch)
        self.attn1 = CBAM(mid_ch + mid_ch)
        
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch)

        self.final_conv = conv1x1(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.init_block(x)

        down_f1 = self.downsample_block1(hx)
        down_f2 = self.downsample_block2(down_f1)

        deep_f = self.deep_conv_block(down_f2)

        deep_f = self.upsample2(deep_f)
        decoded_f2 = torch.cat((down_f1, deep_f[:, :, :, :down_f1.size(3)]), axis=1)
        decoded_f2, _, sa_attn1 = self.attn2(decoded_f2)
        decoded_f2 = self.upsample_features_block2(decoded_f2)

        decoded_f2 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, decoded_f2[:, :, :, :hx.size(3)]), dim=1)
        decoded_f1, _, sa_attn2 = self.attn1(decoded_f1)
        decoded_f1 = self.upsample_features_block1(decoded_f1)

        decoded_f1 = self.final_conv(decoded_f1)

        return decoded_f1, [sa_attn1, sa_attn2]

    
class FourierBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int
    ) -> None:
        super().__init__()
        self.upsample_conv = conv1x1(inplanes, planes)
        self.process = MiniUNet(planes, planes // 2, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.upsample_conv(x)

        out, _ = self.process(identity)

        out += identity
        out = torch.nn.functional.mish(out)

        return out


class FFTCNN(nn.Module):
    def __init__(self, four_normalized=True):
        super().__init__()

        self.four_normalized = 'ortho' if four_normalized else None        
        
        self.basic_layer_1 = FourierBlock(3, 16)
        self.basic_layer_2 = FourierBlock(3, 16)
        # self.final_conv = FourierComplexConv2d(3, 3, 1, 0, use_bias=False)
        self.final_conv1 = nn.Conv2d(16, 3, 1, padding=0, bias=False)
        self.final_conv2 = nn.Conv2d(16, 3, 1, padding=0, bias=False)
            
    def get_fourier(self, x):
        fourier_transform_x = torch.fft.rfft2(
            x, norm=self.four_normalized
        )
        return fourier_transform_x

    def forward(self, image):
        inp = self.get_fourier(image)
        amp_part, angle_part = torch.abs(inp), torch.angle(inp)
        
        x = amp_part
        x = self.basic_layer_1(x)
        x = self.final_conv1(x)

        y = angle_part
        y = self.basic_layer_2(y)
        y = self.final_conv2(y)
        
        x = x * torch.exp(1.j * y)

        restored_x = torch.fft.irfft2(
            x, norm=self.four_normalized
        )

        return restored_x[:, :, :image.size(2), :image.size(3)], []


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.onnx import OperatorExportTypes
    from fvcore.nn import FlopCountAnalysis
    # from pthflops import count_ops
    from pytorch_optimizer import AdaSmooth

    device = 'cuda:0'

    model = FFTCNN().to(device)
    model.apply(init_weights)

    wsize = 512
    img_path = '/media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/cl_img7.jpeg'
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)
    inp = inp.repeat(4, 1, 1, 1)

    with torch.no_grad():
        out = model(inp)

    print('Rand MSE: {}'.format(torch.nn.functional.mse_loss(out[0], inp).item()))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: {} params'.format(params))

    print('TOPS: {:.1f}'.format(FlopCountAnalysis(model, inp).total() / 1E+9))
