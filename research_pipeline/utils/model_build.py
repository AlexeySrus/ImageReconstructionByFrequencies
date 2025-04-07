import torch

from FFTCNN.combined_attn_unet import FFTAttentionUNet
from FFTCNN.combined_attn_unet_plusplus import FFTAttentionUNetPlusPlus
from FFTCNN.uformer import Uformer
from FFTCNN.restormer import Restormer
from FFTCNN.dcnn import DnCNN
from FFTCNN.nafnet import NAFNet
from styleganae.stylegan_v2_models import StyleGANv2AE
from FFTCNN.interpolation_type import interpolation_type_from_str


def build_denoising_model(model_architecture: str, ch_count: int, image_size:int, substracted_noise: bool, interpolation_mode: str, attention_mode:str) -> torch.nn.Module:
    if model_architecture == 'unet':
        model = FFTAttentionUNet(
            in_ch=ch_count,
            out_ch=ch_count,
            image_size=image_size,
            use_substraction=substracted_noise,
            attention_mode=attention_mode,
            interolation_mode=interpolation_type_from_str(interpolation_mode)
        )
    elif model_architecture == 'unetplusplus':
        model = FFTAttentionUNetPlusPlus(
            in_ch=ch_count,
            out_ch=ch_count,
            image_size=image_size,
            use_substraction=substracted_noise,
            attention_mode=attention_mode,
            interolation_mode=interpolation_type_from_str(interpolation_mode)
        )
    elif model_architecture == 'uformer':
        model = Uformer(
            img_size=image_size, embed_dim=32, win_size=8, 
            token_projection='linear', token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True,
            dd_in=ch_count, in_chans=ch_count,
            attention_mode=attention_mode
        )
    elif model_architecture == 'restormer':
        model = Restormer(
            image_size=image_size,
            inp_channels=ch_count,
            out_channels=ch_count,
            attention_mode=attention_mode,
            dim=32,
            LayerNorm_type='BiasFree'
        )
    elif model_architecture == 'dncnn':
        model = DnCNN(in_channels=ch_count, out_channels=ch_count)
    elif model_architecture == 'spatial_stylegan':
        model = StyleGANv2AE(
            z_dim=512,
            c_dim=0,
            w_dim=1024,
            img_resolution=image_size,
            img_channels=ch_count,
            sum_with_style=substracted_noise,
            block_type='spatial'
        )
    elif model_architecture == 'stylegan':
        model = StyleGANv2AE(
            z_dim=512,
            c_dim=0,
            w_dim=1024,
            img_resolution=image_size,
            img_channels=ch_count,
            sum_with_style=substracted_noise,
            block_type='style'
        )
    elif model_architecture == 'nafnet':
        width = 32
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        model = NAFNet(img_channel=ch_count, width=width, middle_blk_num=middle_blk_num,
                       enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    else:
        raise RuntimeError('Unsupported model architecture: {}'.format(model_architecture))
    
    return model
