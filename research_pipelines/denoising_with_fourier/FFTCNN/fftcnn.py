import torch
from torch import nn
import numpy

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
    
    
def conv1x1(inplanes, planes):
    return nn.Conv2d(inplanes, planes, 1, 1, padding=0, dtype=torch.cfloat)
    
    
def conv3x3(inplanes, planes):
    return nn.Conv2d(inplanes, planes, 3, 1, padding=1, dtype=torch.cfloat)


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class FeaturesProcessing(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, in_ch * 2)
        self.act1 = real_imaginary_swish
        self.conv2 = conv3x3(in_ch * 2, out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = real_imaginary_swish

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        _, indices = nn.functional.max_pool2d_with_indices(torch.abs(y), (2, 2))
        y = retrieve_elements_from_indices(y, indices)
        return y
    

class CustomUpsampleNearest2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y):
        new_y = torch.zeros(y.size(0), y.size(1), y.size(2) * 2, y.size(3) * 2, dtype=y.dtype, device=y.device)

        for k in range(2):
            for q in range(2):
                new_y[:, :, k::2, q::2] = torch.clone(y)

        return new_y
    

class FeaturesUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = CustomUpsampleNearest2x()
        self.features = FeaturesProcessing(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)
        y = self.features(y)
        y = real_imaginary_swish(y)
        return y


class MiniUNet(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch)

        self.upsample2 = FeaturesUpsample(mid_ch, mid_ch)
        self.upsample1 = FeaturesUpsample(mid_ch, mid_ch)
        
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
        decoded_f2 = self.upsample_features_block2(decoded_f2)

        decoded_f2 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, decoded_f2[:, :, :, :hx.size(3)]), dim=1)
        decoded_f1 = self.upsample_features_block1(decoded_f1)

        decoded_f1 = self.final_conv(decoded_f1)

        return decoded_f1, []

    
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
        out = real_imaginary_swish(out)

        return out


class FFTCNN(nn.Module):
    def __init__(self, four_normalized=True):
        super().__init__()

        self.four_normalized = 'ortho' if four_normalized else None        
        
        self.basic_layer_1 = FourierBlock(3, 16)
        self.final_comples_conv = nn.Conv2d(16, 3, 1, dtype=torch.cfloat)
        self.final_real_conv = nn.Conv2d(3, 3, 1, dtype=torch.float32)
            
    def get_fourier(self, x):
        fourier_transform_x = torch.fft.fft2(
            x, norm=self.four_normalized
        )
        return fourier_transform_x

    def forward(self, image):
        inp = self.get_fourier(image)
        
        x = self.basic_layer_1(inp)
        x = self.final_comples_conv(x)

        x = inp - x

        restored_x = torch.fft.ifft2(
            x, norm=self.four_normalized
        )

        restored_x = self.final_real_conv(torch.abs(restored_x))

        return restored_x, []


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.onnx import OperatorExportTypes

    device = 'cpu'

    model = FFTCNN().to(device)
    model.apply(init_weights)

    wsize = 512
    img_path = '/Users/alexey/Downloads/st_photo_1.jpg'
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)
    inp = inp.repeat(4, 1, 1, 1)

    with torch.no_grad():
        out, _ = model(inp)


    print('Rand L1: {}'.format(torch.nn.functional.l1_loss(out, inp).item()))
