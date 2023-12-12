from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn

from WTSNet.attention import CBAM
from utils.haar_utils import HaarForward, HaarInverse


padding_mode: str = 'reflect'


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


def convert_weights_from_old_version(_weights: OrderedDict) -> OrderedDict:
    new_weights = OrderedDict(
        [('wtsmodel.' + k, v) if not k.startswith('wtsmodel.') else (k, v) for k, v in _weights.items()]
    )
    return new_weights


class DownscaleByWaveletes(nn.Module):
    def __init__(self) -> None:
        """
        Downscale by Haar Wavelets transforms
        This transform increase channels by 4 times: C -> C * 4
        """
        super().__init__()

        self.dwt = HaarForward()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.dwt(x)
        step = out.size(1) // 4
        return out[:, :step], out[:, step:]


class UpscaleByWaveletes(nn.Module):
    def __init__(self) -> None:
        """
        Downscale by Haar Wavelets transforms
        This transform reduce channels by 4 times: C -> C // 4
        """
        super().__init__()

        self.iwt = HaarInverse()

    def forward(self, x_ll: torch.Tensor, hight_freq: torch.Tensor) -> torch.Tensor:
        out = self.iwt(torch.cat((x_ll, hight_freq), dim=1))
        return out
    

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
        self.act1 = nn.ReLU()
        self.conv2 = conv3x3(in_ch * 2, out_ch)

        self.down_bneck = conv1x1(in_ch, out_ch)

        self.act_final = nn.ReLU()

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
        self.pool = nn.MaxPool2d(2, 2)

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
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, need_up_features: bool = False):
        super().__init__()
        self.init_block = FeaturesProcessing(in_ch, mid_ch)

        self.downsample_block1 = FeaturesDownsample(mid_ch, mid_ch)
        self.downsample_block2 = FeaturesDownsample(mid_ch, mid_ch * 2)

        self.deep_conv_block = FeaturesProcessing(mid_ch * 2, mid_ch)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2) # FeaturesUpsample(mid_ch, mid_ch)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2) # FeaturesUpsample(mid_ch, mid_ch)
        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.init_attn = CBAM(mid_ch)
        self.attn2 = CBAM(mid_ch + mid_ch)
        self.attn1 = CBAM(mid_ch + mid_ch)
        
        self.upsample_features_block2 = FeaturesProcessing(mid_ch + mid_ch, mid_ch)
        self.upsample_features_block1 = FeaturesProcessing(mid_ch + mid_ch, out_ch)

        self.final_conv = conv1x1(out_ch, out_ch)

        self.conv_out = FeaturesProcessingWithLastConv(out_ch, out_ch)
        self.up_conv_out = FeaturesProcessingWithLastConv(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.init_block(x)
        hx, _, sa_attn0 = self.init_attn(hx)

        down_f1 = self.downsample_block1(hx)
        down_f2 = self.downsample_block2(down_f1)

        deep_f = self.deep_conv_block(down_f2)

        deep_f = self.upsample2(deep_f)
        decoded_f2 = torch.cat((down_f1, deep_f), axis=1)
        decoded_f2, _, sa_attn1 = self.attn2(decoded_f2)
        decoded_f2 = self.upsample_features_block2(decoded_f2)

        decoded_f2 = self.upsample1(decoded_f2)
        decoded_f1 = torch.cat((hx, decoded_f2), dim=1)
        decoded_f1, _, sa_attn2 = self.attn1(decoded_f1)
        decoded_f1 = self.upsample_features_block1(decoded_f1)

        upf1 = self.upsample0(decoded_f1)
        upf1 = self.up_conv_out(upf1)

        decoded_f1 = self.final_conv(decoded_f1)

        return decoded_f1, upf1, [sa_attn0, sa_attn1, sa_attn2]
    

class MiniUNetV2(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, need_up_features: bool = False):
        super().__init__()
        self.need_up_features = need_up_features

        self.rebnconvin = FeaturesProcessingWithLastConv(in_ch, out_ch)

        self.rebnconv1 = FeaturesProcessingWithLastConv(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.rebnconv2 = FeaturesProcessingWithLastConv(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.rebnconv3 = FeaturesProcessingWithLastConv(mid_ch, mid_ch)

        self.rebnconv4 = FeaturesProcessingWithLastConv(mid_ch, mid_ch)

        self.rebnconv3d = FeaturesProcessingWithLastConv(mid_ch*2, mid_ch)
        self.rebnconv2d = FeaturesProcessingWithLastConv(mid_ch*2,mid_ch)
        self.rebnconv1d = FeaturesProcessingWithLastConv(mid_ch*2, out_ch)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.rebnconvout = FeaturesProcessingWithLastConv(out_ch, out_ch)

        self.init_attn = CBAM(out_ch)
        self.attn3 = CBAM(mid_ch + mid_ch)
        self.attn2 = CBAM(mid_ch + mid_ch)
        self.attn1 = CBAM(mid_ch + mid_ch)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hxin, _, sa_attn0 = self.init_attn(hxin)


        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx4_hx3 = torch.cat((hx4, hx3), dim=1)
        hx4_hx3, _, sa_attn3 = self.attn3(hx4_hx3)
        hx3d = self.rebnconv3d(hx4_hx3)
        hx3dup = self.upsample2(hx3d)

        hx3dup_hx2 = torch.cat((hx3dup, hx2), dim=1)
        hx3dup_hx2, _, sa_attn2 = self.attn2(hx3dup_hx2)
        hx2d = self.rebnconv2d(hx3dup_hx2)
        hx2dup = self.upsample1(hx2d)

        hx2dup_hx1 = torch.cat((hx2dup, hx1), dim=1)
        hx2dup_hx1, _, sa_attn1 = self.attn1(hx2dup_hx1)
        hx1d = self.rebnconv1d(hx2dup_hx1)

        if self.need_up_features:
            out_features = self.rebnconvout(self.upsample0(hx1d))
        else:
            out_features = hx1d

        return hx1d + hxin, out_features, [sa_attn0, sa_attn1, sa_attn2, sa_attn3]


class WTSNetBaseModel(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()

        self.low_freq_to_wavelets_f1 = FeaturesProcessingWithLastConv(image_channels, 64)
        self.hight_freq_u1 = MiniUNet(64 + image_channels * 3 + 64, 64, 128)
        self.hight_freq_c1 = conv1x1(128, image_channels * 3)

        self.low_freq_to_wavelets_f2 = FeaturesProcessingWithLastConv(image_channels, 64)
        self.hight_freq_u2 = MiniUNet(64 + image_channels * 3 + 64, 64, 64, True)
        self.hight_freq_c2 = conv1x1(64, image_channels * 3)

        self.low_freq_to_wavelets_f3 = FeaturesProcessingWithLastConv(image_channels, 32)
        self.hight_freq_u3 = MiniUNet(32 + image_channels * 3 + 32 + 32, 32, 64, True)
        self.hight_freq_c3 = conv1x1(64, image_channels * 3)

        self.low_freq_u4 = MiniUNetV2(image_channels, 16, 32, True)
        self.low_freq_c4 = conv1x1(32, image_channels)
        self.low_freq_to_wavelets_f4 = FeaturesProcessingWithLastConv(image_channels, 32)
        self.hight_freq_u4 = MiniUNet(32 + image_channels * 3, 16, 32, True)
        self.hight_freq_c4 = conv1x1(32, image_channels * 3)

    def forward(self, ll_list: List[torch.Tensor], hf_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ll1, ll2, ll3, ll4 = ll_list
        hf1, hf2, hf3, hf4 = hf_list

        t4_wf = self.low_freq_to_wavelets_f4(ll4 - 1)
        df_hd4, hd4up, sa4 = self.hight_freq_u4(
            torch.cat((t4_wf, hf4), dim=1)
        )
        df_hd4 = self.hight_freq_c4(df_hd4)
        hf4 -= df_hd4
        t4, t4up, sa5 = self.low_freq_u4(ll4 - 1)
        pred_ll4 = self.low_freq_c4(t4) + 1
        # pred_ll4 = ll4 - df_pred_ll4

        t3_wf = self.low_freq_to_wavelets_f3(ll3 - 1)
        df_hd3, hd3up, sa3 = self.hight_freq_u3(
            torch.cat((t3_wf, hf3, t4up, hd4up), dim=1)
        )
        df_hd3 = self.hight_freq_c3(df_hd3)
        hf3 -= df_hd3

        t2_wf = self.low_freq_to_wavelets_f2(ll2 - 1)
        df_hd2, hd2up, sa2 = self.hight_freq_u2(
            torch.cat((t2_wf, hf2, hd3up), dim=1)
        )
        df_hd2 = self.hight_freq_c2(df_hd2)
        hf2 -= df_hd2

        t1_wf = self.low_freq_to_wavelets_f1(ll1 - 1)
        df_hd1, _, sa1 = self.hight_freq_u1(
            torch.cat((t1_wf, hf1, hd2up), dim=1)
        )
        df_hd1 = self.hight_freq_c1(df_hd1)
        hf1 -= df_hd1

        return hf1, hf2, hf3, hf4, pred_ll4, [sa1, sa2, sa3, sa4, sa5]


class WTSNet(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()

        self.dwt1 = DownscaleByWaveletes()
        self.dwt2 = DownscaleByWaveletes()
        self.dwt3 = DownscaleByWaveletes()
        self.dwt4 = DownscaleByWaveletes()

        self.wtsmodel = WTSNetBaseModel(image_channels)

        self.iwt1 = UpscaleByWaveletes()
        self.iwt2 = UpscaleByWaveletes()
        self.iwt3 = UpscaleByWaveletes()
        self.iwt4 = UpscaleByWaveletes()


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ll1, hf1 = self.dwt1(x)
        ll2, hf2 = self.dwt2(ll1 / 2)
        ll3, hf3 = self.dwt3(ll2 / 2)
        ll4, hf4 = self.dwt4(ll3 / 2)

        hf1, hf2, hf3, hf4, pred_ll4, sa_list = self.wtsmodel(
            [ll1, ll2, ll3, ll4],
            [hf1, hf2, hf3, hf4]
        )

        pred_ll3 = self.iwt4(pred_ll4, hf4) * 2
        pred_ll2 = self.iwt3(pred_ll3, hf3) * 2
        pred_ll1 = self.iwt2(pred_ll2, hf2) * 2
        pred_image = self.iwt1(pred_ll1, hf1)

        wavelets1 = torch.cat((pred_ll1, hf1), dim=1)
        wavelets2 = torch.cat((pred_ll2, hf2), dim=1)
        wavelets3 = torch.cat((pred_ll3, hf3), dim=1)
        wavelets4 = torch.cat((pred_ll4, hf4), dim=1)

        sa1, sa2, sa3, sa4, sa5 = sa_list
        sa1 = nn.functional.interpolate(sa1[0], (x.size(2), x.size(3)), mode='area')
        sa2 = nn.functional.interpolate(sa2[0], (x.size(2), x.size(3)), mode='area')
        sa3 = nn.functional.interpolate(sa3[0], (x.size(2), x.size(3)), mode='area')
        sa4 = nn.functional.interpolate(sa4[0], (x.size(2), x.size(3)), mode='area')
        sa5 = nn.functional.interpolate(sa5[0], (x.size(2), x.size(3)), mode='area')


        return pred_image, [wavelets1, wavelets2, wavelets3, wavelets4], [sa1, sa2, sa3, sa4, sa5]


if __name__ == '__main__':
    import cv2
    import numpy as np
    from torch.onnx import OperatorExportTypes
    from fvcore.nn import FlopCountAnalysis
    # from pthflops import count_ops
    from pytorch_optimizer import AdaSmooth

    class DWTHaar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dwt = HaarForward()

        def forward(self, x):
            out = self.dwt(x)
            step = out.size(1) // 4
            ll = out[:, :step]
            lh = out[:, step:step*2]
            hl = out[:, step*2:step*3]
            hh = out[:, step*3:]
            return [ll, lh, hl, hh]

    device = 'cuda:0'

    model = WTSNet().to(device)
    model.apply(init_weights)

    wsize = 512
    img_path = '/media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/cl_img7.jpeg'
    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)
    inp[...] = 0.5

    haar = HaarForward()
    ihaar = HaarInverse()

    wavelets = haar(inp)
    img_reconstructed = ihaar(wavelets)
    step = wavelets.size(1) // 4
    print(step, wavelets.shape)
    ll = wavelets[:, :step] # :3
    lh = wavelets[:, step:step*2] # 3:6
    hl = wavelets[:, step*2:step*3] # 6:9
    hh = wavelets[:, step*3:] # 9:12
    
    for h, n in zip((ll, lh, hl, hh), ('ll', 'lh', 'hl', 'hh')):
        print('Haar part: {}: {} {} {}'.format(n, h.min(), h.max(), h.shape))

    with torch.no_grad():
        out = model(inp)

    print('Rand MSE: {}'.format(torch.nn.functional.mse_loss(out[0], inp).item()))

    for block_output, block_name in zip(out[1], ('d{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    print()

    for block_output, block_name in zip(out[2], ('sa_{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Params: {} params'.format(params))

    print('TFOPS: {:.1f}'.format(FlopCountAnalysis(model, inp).total() / 1E+9))

    # torch.onnx.export(model,               # model being run
    #                 inp,                         # model input (or a tuple for multiple inputs)
    #                 "haar.onnx",   # where to save the model (can be a file or file-like object)
    #                 export_params=True,        # store the trained parameter weights inside the model file
    #                 opset_version=16,          # the ONNX version to export the model to
    #                 do_constant_folding=True,  # whether to execute constant folding for optimization
    #                 input_names = ['input'],   # the model's input names
    #                 output_names = ['output'], # the model's output names
    #                 dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3: 'width'},    # variable length axes
    #                                 'output' : {0 : 'batch_size', 2: 'height', 3: 'width'}},
    #                 operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
    dwt = DWTHaar()

    def compute_wavelets_loss(pred_wavelets_pyramid, gt_image, factor: float = 0.8):
        gt_d0_ll = gt_image
        _loss = 0.0
        _loss_scale = 1.0

        for i in range(len(pred_wavelets_pyramid)):
            gt_ll, gt_lh, gt_hl, gt_hh = dwt(gt_d0_ll)

            if i < len(pred_wavelets_pyramid) - 1:
                gt_wavelets = torch.cat((gt_ll, gt_lh, gt_hl, gt_hh), dim=1)
                _loss += torch.nn.functional.smooth_l1_loss(pred_wavelets_pyramid[i], gt_wavelets) * _loss_scale
            else:
                gt_wavelets = torch.cat((gt_lh, gt_hl, gt_hh), dim=1)
                _loss += torch.nn.functional.smooth_l1_loss(pred_wavelets_pyramid[i][:, 3:], gt_wavelets) * _loss_scale

            _loss_scale *= factor

            gt_d0_ll = gt_ll / 2.0

        return _loss


    # model2 = model
    # optim = torch.optim.SGD(params=model2.parameters(), lr=0.1, weight_decay=0.00001, nesterov=True, momentum=0.9)

    # res_img = torch.clamp(inp, 0, 1).to('cpu')[0].permute(1, 2, 0).numpy()
    # res_img = (res_img * 255).astype(np.uint8)[:, :, ::-1]
    # cv2.imwrite('TEST_IMG_GT.png', res_img)

    # N = 5000
    # for i in range(N):
    #     optim.zero_grad()
    #     pred = model2(inp)
    #     px_loss = torch.nn.functional.mse_loss(pred[0], inp)
    #     w_loss = compute_wavelets_loss(pred[1], inp)
    #     loss = px_loss + w_loss
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model2.parameters(), 2.0)
    #     optim.step()

    #     if (i + 1) % 100 == 0:
    #         print('Step {}/{}, Loss: {}'.format(i + 1, N, px_loss.item()))
    #         res_img = torch.clamp(pred[0].detach(), 0, 1).to('cpu')[0].permute(1, 2, 0).numpy()
    #         res_img = (res_img * 255).astype(np.uint8)[:, :, ::-1]
    #         cv2.imwrite('TEST_IMG.png', res_img)
