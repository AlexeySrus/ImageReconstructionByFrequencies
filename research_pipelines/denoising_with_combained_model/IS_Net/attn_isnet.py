import torch
import torch.nn as nn

from IS_Net.cbam import CBAM
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


padding_mode: str = 'reflect'


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride, padding_mode=padding_mode)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):

    src = nn.functional.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


class DownscaleByWaveletes(nn.Module):
    def __init__(self, in_ch: int, wavename: str = 'haar') -> None:
        super().__init__()

        self.dwt = DWT_2D(wavename)

        self.reparam_conv = nn.Conv2d(in_ch * 4, in_ch, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll, x_lh, x_hl, x_hh = self.dwt(x)
        out = torch.cat((x_ll, x_lh, x_hl, x_hh), dim=1)
        out = self.reparam_conv(out)
        return out


class UpscaleByWaveletes(nn.Module):
    def __init__(self, in_ch: int, wavename: str = 'haar') -> None:
        super().__init__()

        self.iwt = IDWT_2D(wavename)

        self.reparam_conv = nn.Conv2d(in_ch // 4, in_ch, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll, x_lh, x_hl, x_hh = torch.split(x, x.size(1) // 4, dim=1)
        out = self.iwt(x_ll, x_lh, x_hl, x_hh)
        out = self.reparam_conv(out)
        return out


### RSU-7 ###
class RSU7(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512, use_attention: bool = False):
        super(RSU7,self).__init__()
        self.use_attention = use_attention

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1) ## 1 -> 1/2
        if self.use_attention:
            self.attention_layer = CBAM(in_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        b, c, h, w = x.shape

        hx = x

        if self.use_attention:
            hx, _, sa_hx = self.attention_layer(hx)
        else:
            sa_hx = hx

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin, sa_hx
    

### RSU-6 ###
class RSU6(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_attention: bool = False):
        super(RSU6,self).__init__()
        self.use_attention = use_attention

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        if self.use_attention:
            self.attention_layer = CBAM(in_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        if self.use_attention:
            hx, _, sa_hx = self.attention_layer(hx)
        else:
            sa_hx = hx

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin, sa_hx

### RSU-5 ###
class RSU5(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_attention: bool = False):
        super(RSU5,self).__init__()
        self.use_attention = use_attention

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        if self.use_attention:
            self.attention_layer = CBAM(in_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        if self.use_attention:
            hx, _, sa_hx = self.attention_layer(hx)
        else:
            sa_hx = hx

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin, sa_hx

### RSU-4 ###
class RSU4(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_attention: bool = False):
        super(RSU4,self).__init__()
        self.use_attention = use_attention

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        if self.use_attention:
            self.attention_layer = CBAM(in_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        if self.use_attention:
            hx, _, sa_hx = self.attention_layer(hx)
        else:
            sa_hx = hx

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin, sa_hx

### RSU-4F ###
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_attention: bool = False):
        super(RSU4F,self).__init__()
        self.use_attention = use_attention

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        if self.use_attention:
            self.attention_layer = CBAM(in_ch)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        if self.use_attention:
            hx, _, sa_hx = self.attention_layer(hx)
        else:
            sa_hx = hx

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin, sa_hx


class myrebnconv(nn.Module):
    def __init__(self, in_ch=3,
                       out_ch=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       dilation=1,
                       groups=1):
        super(myrebnconv,self).__init__()

        self.conv = nn.Conv2d(in_ch,
                              out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.rl(self.bn(self.conv(x)))

    
class ISNetDIS(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, image_ch=3):
        super(ISNetDIS,self).__init__()

        self.enable_upscale: bool = False

        self.conv_in = nn.Conv2d(in_ch,64,3,stride=1,padding=1, padding_mode=padding_mode)

        # self.attn_s1 = CBAM(64)
        self.stage1 = RSU7(64,32,64, use_attention=False)
        self.pool12 = DownscaleByWaveletes(64)

        # self.attn_s2 = CBAM(64)
        self.stage2 = RSU6(64,32,128, use_attention=False)
        self.pool23 = DownscaleByWaveletes(128)

        # self.attn_s3 = CBAM(128)
        self.stage3 = RSU5(128,64,256, use_attention=False)
        self.pool34 =  DownscaleByWaveletes(256)

        # self.attn_s4 = CBAM(256)
        self.stage4 = RSU4(256,128,512, use_attention=False)
        self.pool45 = DownscaleByWaveletes(512)

        # self.attn_s5 = CBAM(512)
        self.stage5 = RSU4F(512,256,512, use_attention=False)
        self.pool56 = DownscaleByWaveletes(512)

        # self.attn_s6 = CBAM(512)
        self.stage6 = RSU4F(512,256,512, use_attention=True)

        # decoder
        self.stage5d = RSU4F(1024,256,512, use_attention=True)
        self.stage4d = RSU4(1024,128,256, use_attention=True)
        self.stage3d = RSU5(512,64,128, use_attention=True)
        self.stage2d = RSU6(256,32,64, use_attention=True)
        self.stage1d = RSU7(128,16,64, use_attention=True)

        self.up6 = UpscaleByWaveletes(512)
        self.up5 = UpscaleByWaveletes(512)
        self.up4 = UpscaleByWaveletes(256)
        self.up3 = UpscaleByWaveletes(128)
        self.up2 = UpscaleByWaveletes(64)

        self.side1 = nn.Conv2d(64,image_ch,3,padding=1, padding_mode=padding_mode)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1, padding_mode=padding_mode)

    def forward(self, x):
        # Normilize from 0..1 to -1..1
        x = (x - 0.5) * 2

        hx = x

        hxin = self.conv_in(hx)

        #stage 1
        hx1, _ = self.stage1(hxin)
        hx = self.pool12(hx1)

        #stage 2
        hx2, _ = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3, _ = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4, _ = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5, _ = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6, sa_hx6 = self.stage6(hx)
        hx6up = self.up6(hx6)

        #-------------------- decoder --------------------
        hx5d, sa_hx5 = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = self.up5(hx5d)

        hx4d, sa_hx4 = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = self.up4(hx4d)

        hx3d, sa_hx3 = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = self.up3(hx3d)

        hx2d, sa_hx2 = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = self.up2(hx2d)

        hx1d, sa_hx1 = self.stage1d(torch.cat((hx2dup,hx1),1))

        #side output
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        # Denormilize RGB channels of image from -1..1 to 0..1
        d1[:, :3] = d1[:, :3] / 2 + 0.5

        # Denormilize only RGB channels of Wavelets outputs from -1..1 to 0..2
        d2[:, :3] = d2[:, :3] + 1
        d3[:, :3] = d3[:, :3] + 1
        d4[:, :3] = d4[:, :3] + 1
        d5[:, :3] = d5[:, :3] + 1
        d6[:, :3] = d6[:, :3] + 1

        sa_hx2 = _upsample_like(sa_hx2, x)
        sa_hx3 = _upsample_like(sa_hx3, x)
        sa_hx4 = _upsample_like(sa_hx4, x)
        sa_hx5 = _upsample_like(sa_hx5, x)
        sa_hx6 = _upsample_like(sa_hx6, x)

        return [d1, d2, d3, d4, d5, d6], [sa_hx1, sa_hx2, sa_hx3, sa_hx4, sa_hx5, sa_hx6]


if __name__ == '__main__':
    import numpy as np

    device = 'cpu'

    model = ISNetDIS().to(device)
    inp = torch.rand(1, 3, 512, 512).to(device)

    with torch.no_grad():
        out = model(inp)

    for block_output, block_name in zip(out[0], ('d{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    print()

    for block_output, block_name in zip(out[1], ('sa_hx{}d'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: {} params'.format(params))
