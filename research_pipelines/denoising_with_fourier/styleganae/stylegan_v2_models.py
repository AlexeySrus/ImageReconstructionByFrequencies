# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

from typing import List

import numpy as np
import torch
from collections import OrderedDict

try:
    from .torch_utils import misc
    from .torch_utils import persistence
    from .torch_utils.ops import conv2d_resample
    from .torch_utils.ops import upfirdn2d
    from .torch_utils.ops import bias_act
    from .torch_utils.ops import fma
except:
    from torch_utils import misc
    from torch_utils import persistence
    from torch_utils.ops import conv2d_resample
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import fma

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        assert in_features is not None and in_features > 0
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])
    

class CameraInfoMLP(torch.nn.Module):
    def __init__(self, emb_dim, h_dim=32):
        super().__init__()

        self.cam_model_embedder = torch.nn.Embedding(num_embeddings=5, embedding_dim=h_dim)
        self.light_intensity_embedder = torch.nn.Embedding(num_embeddings=3, embedding_dim=h_dim)
        self.light_temperature_embedder = torch.nn.Embedding(num_embeddings=38, embedding_dim=h_dim)
        self.iso_embedder = FullyConnectedLayer(1, h_dim)
        self.shutter_embedding = FullyConnectedLayer(1, h_dim)

        self.emb_dim = emb_dim
        self.cond_emb = torch.nn.Sequential(
            FullyConnectedLayer(h_dim * 5, emb_dim),
            torch.nn.SELU(),
            FullyConnectedLayer(emb_dim, emb_dim)
        )

    def forward(self, iso, shutter, light_temp, cam, light_intensity):
        iso_emb = self.iso_embedder(iso)
        shutter_emb = self.shutter_embedding(shutter)
        k_emb = self.light_temperature_embedder(light_temp).squeeze(1)
        cam_emb = self.cam_model_embedder(cam).squeeze(1)
        li_emb = self.light_intensity_embedder(light_intensity).squeeze(1)

        emb = torch.concat(
            (
                iso_emb,
                shutter_emb,
                k_emb,
                cam_emb,
                li_emb
            ),
            dim=1
        )
        return self.cond_emb(emb)
    

class CameraInfoWithModeMLP(torch.nn.Module):
    def __init__(self, emb_dim, h_dim=32):
        super().__init__()

        self.cam_model_embedder = torch.nn.Embedding(num_embeddings=5, embedding_dim=h_dim)
        self.light_intensity_embedder = torch.nn.Embedding(num_embeddings=3, embedding_dim=h_dim)
        self.light_temperature_embedder = torch.nn.Embedding(num_embeddings=38, embedding_dim=h_dim)
        self.iso_embedder = FullyConnectedLayer(1, h_dim)
        self.shutter_embedding = FullyConnectedLayer(1, h_dim)
        self.mode = torch.nn.Embedding(num_embeddings=2, embedding_dim=h_dim)

        self.emb_dim = emb_dim
        self.cond_emb = torch.nn.Sequential(
            FullyConnectedLayer(h_dim * 6, emb_dim),
            torch.nn.SELU(),
            FullyConnectedLayer(emb_dim, emb_dim)
        )

    def forward(self, iso, shutter, light_temp, cam, light_intensity, mode):
        iso_emb = self.iso_embedder(iso)
        shutter_emb = self.shutter_embedding(shutter)
        k_emb = self.light_temperature_embedder(light_temp).squeeze(1)
        cam_emb = self.cam_model_embedder(cam).squeeze(1)
        li_emb = self.light_intensity_embedder(light_intensity).squeeze(1)
        mode_emb = self.mode(mode).squeeze(1)

        emb = torch.concat(
            (
                iso_emb,
                shutter_emb,
                k_emb,
                cam_emb,
                li_emb,
                mode_emb
            ),
            dim=1
        )
        return self.cond_emb(emb)
    

class FlattenAfterPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2).squeeze(2)


class SpatialAttentionPooling(torch.nn.Module):
    def __init__(self, in_channels, out_dim, lr_multiplier=0.1):
        super().__init__()
        self.attention = Conv2dLayer(in_channels, in_channels, 1, activation="sigmoid")
        self.proj = FullyConnectedLayer(in_channels, out_dim, lr_multiplier=lr_multiplier)

    def forward(self, x):
        attn = self.attention(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x * attn, (1, 1)).squeeze(3).squeeze(2)
        return self.proj(pooled) 

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                if c is not None:
                    misc.assert_shape(c, [None, self.c_dim])
                else:
                    c = torch.zeros(z.size(0), self.c_dim, device=z.device)
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = False,        # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights
        down            = 1             # For downsampling
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.down = down
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            # self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            self.noise_strength = FullyConnectedLayer(w_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength(w).unsqueeze(1).unsqueeze(2)
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength(w).unsqueeze(1).unsqueeze(2)

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up, down=self.down,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SpatialSynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)
            
        self.injection_layers = []
        if in_channels != 0:
            for _ in range(1):
                injection_layer = Conv2dLayer(out_channels, out_channels, 1)
                self.injection_layers.append(injection_layer)

        if len(self.injection_layers) > 0:
            self.injection_layers = torch.nn.ModuleList(self.injection_layers)
        else:
            self.injection_layers = None
            
    def apply_injection(self, x: torch.Tensor, encoder_features: torch.Tensor, injection_layer_idx: torch.nn.Module) -> torch.Tensor:
        if self.injection_layers is None:
            return x

        return torch.nn.functional.leaky_relu(x + self.injection_layers[injection_layer_idx](encoder_features), 0.2)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, style_features=None, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        if style_features is not None:
            sf = style_features.pop(0)

        # Input.
        if self.in_channels == 0:
            x = sf.to(dtype=dtype, memory_format=memory_format)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))

            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.apply_injection(x, sf, 0)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)

            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.apply_injection(x, sf, 0)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
    

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)
            
        self.injection_layers = []

        if in_channels != 0:
            for _ in range(self.num_conv + self.num_torgb):
                inject_layer = SpatialAttentionPooling(out_channels, w_dim)
                self.injection_layers.append(inject_layer)

        if len(self.injection_layers) > 0:
            self.injection_layers = torch.nn.ModuleList(self.injection_layers)
        else:
            self.injection_layers = None
            
            
    def apply_injection(self, w_features: torch.Tensor, style_features: List[torch.Tensor]) -> torch.Tensor:
        if style_features is None:
            return w_features

        injected_w = style_features.pop(0)

        return w_features + injected_w

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, style_features=None, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        encoder_injection_idx = 0
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = style_features.pop(0)
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, self.apply_injection(next(w_iter), style_features), fused_modconv=fused_modconv, **layer_kwargs)
            encoder_injection_idx += 1
            x = self.conv1(x, self.apply_injection(next(w_iter), style_features), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, self.apply_injection(next(w_iter), style_features), fused_modconv=fused_modconv, **layer_kwargs)
            encoder_injection_idx += 1
            x = self.conv1(x, self.apply_injection(next(w_iter), style_features), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            encoder_injection_idx += 1
            y = self.torgb(x, self.apply_injection(next(w_iter), style_features), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_resolution,                 # Output image resolution.
        img_channels,                   # Number of color channels.
        channel_base    = 32768,        # Overall multiplier for the number of channels.
        channel_max     = 512,          # Maximum number of channels in any layer.
        num_fp16_res    = 4,            # Use FP16 for the N highest resolutions.
        block_type      = 'spatial',    # Technique to use encoder features (spatial or style).
        **block_kwargs,                 # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert block_type in ['spatial', 'style']
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        synthesis_block_class = SpatialSynthesisBlock if block_type == 'spatial' else SynthesisBlock

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = synthesis_block_class(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, style_features=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, style_features=style_features, **block_kwargs)
            
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        without_downsample  = False,
        embedding_channels  = None,
        use_noise           = False
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.without_downsample = without_downsample
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        self.down = (1 if self.without_downsample else 2)

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=self.down,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=self.down,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False, emb=None, ws=None):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter, down=self.down) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype

        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
    

@persistence.persistent_class
class EncoderBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        without_downsample  = False,
        embedding_channels  = None,
        use_noise           = False
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.embedding_channels = embedding_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.without_downsample = without_downsample
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        self.down = (1 if self.without_downsample else 2)

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = SynthesisLayer(img_channels, tmp_channels, w_dim=embedding_channels, kernel_size=1, activation=activation,
                conv_clamp=conv_clamp, channels_last=self.channels_last, resolution=resolution)
            self.num_layers += 1

        self.conv0 = SynthesisLayer(tmp_channels, tmp_channels, w_dim=embedding_channels, kernel_size=3, activation=activation,
            conv_clamp=conv_clamp, channels_last=self.channels_last, resolution=resolution)
        self.num_layers += 1

        self.conv1 = SynthesisLayer(tmp_channels, out_channels, w_dim=embedding_channels, kernel_size=3, activation=activation, down=self.down,
            resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, resolution=resolution)
        self.num_layers += 1

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=self.down,
                trainable=True, resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False, ws=None):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        ws_iter = iter(ws.unbind(dim=1))

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img, next(ws_iter))
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter, down=self.down) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(ws_iter))
            x = self.conv1(x, next(ws_iter), gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x, next(ws_iter))
            x = self.conv1(x, next(ws_iter))

        assert x.dtype == dtype

        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
    


@persistence.persistent_class
class UNetEncoderBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        without_downsample  = False,
        embedding_channels  = None,
        use_noise           = False
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.without_downsample = without_downsample
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        self.down = (1 if self.without_downsample else 2)

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=self.down,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=self.down,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)
        
        self.emb_inject1 = None
        self.emb_inject2 = None
        if embedding_channels is not None:
            self.emb_inject1 = FullyConnectedLayer(embedding_channels, tmp_channels)
            self.emb_inject2 = FullyConnectedLayer(embedding_channels, out_channels)

    def inject_embedding(self, x, emb, layer):
        if emb is None or layer is None:
            return x
        
        enc_emb = layer(torch.nn.functional.leaky_relu(emb, 0.2)).view(x.size(0), x.size(1), 1, 1)
        return x + enc_emb

    def forward(self, x, img, force_fp32=False, emb=None, ws=None):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter, down=self.down) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.inject_embedding(x, emb, self.emb_inject1)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = self.inject_embedding(x, emb, self.emb_inject2)
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.inject_embedding(x, emb, self.emb_inject1)
            x = self.conv1(x)
            x = self.inject_embedding(x, emb, self.emb_inject2)

        assert x.dtype == dtype

        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
    

@persistence.persistent_class
class PathGANDiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.final_conv = Conv2dLayer(in_channels, 1, kernel_size=3)
       

    def forward(self, x, img, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.final_conv(x)

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img, _ = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
    

class CoditionalModule(object):
    def calculate_embedding(self, t, condition):
        emb = None

        if self.time_embedder is not None and t is not None:
            emb = self.time_embedder(t)

        if condition is not None and self.cond_embedder is not None:
            cond_emb = self.cond_embedder(**condition)

            if emb is not None:
                emb += cond_emb
            else:
                emb = cond_emb

        return emb


class ConditionalDiscriminator(torch.nn.Module, CoditionalModule):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        use_pathgan         = False,    # Use as PathGAN discriminator
        encode_embedding    = False,
        time_embedder: torch.nn.Module = None, time_embedder_kwargs={}, cond_embedder: torch.nn.Module = None, cond_embedder_kwargs={}
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.use_pathgan = use_pathgan
        self.encode_embedding = encode_embedding
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if time_embedder is not None:
            self.time_embedder = time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None

        if cond_embedder is not None:
            self.cond_embedder = cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None
            cond_emb_dim = None

        if time_emb_dim is None and cond_emb_dim is not None:
            time_emb_dim = cond_emb_dim

        c_dim = time_emb_dim if time_emb_dim is not None else 0
        self.c_dim = c_dim

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

            if time_emb_dim is not None and (use_pathgan or encode_embedding):
                latent_embedder = SynthesisLayer(out_channels, out_channels, time_emb_dim, res // 2, use_noise=False)
                setattr(self, f'e{res}', latent_embedder)

        if use_pathgan:
            self.b4 = PathGANDiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        else:
            if c_dim > 0:
                self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

            self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

        self.unconditional_embedding = None
        if time_emb_dim is not None:
            self.unconditional_embedding = torch.nn.Parameter(torch.randn(1, time_emb_dim))

    def compute_features_loss(self, img1, img2, t=None, condition=None):
        return torch.nn.functional.l1_loss(
            self.forward(img1, t=t, condition=condition, for_features_matching=True).view(img1.size(0), -1),
            self.forward(img2, t=t, condition=condition, for_features_matching=True).view(img2.size(0), -1)
        )

    def forward(self, img, t=None, condition=None, update_emas=False, for_features_matching=False, select_mask=None, **block_kwargs):
        emb = self.calculate_embedding(t, condition)

        if select_mask is not None:
            uncond_count = img.size(0) - int(select_mask.sum().item())
            if uncond_count > 0:
                emb[torch.bitwise_not(select_mask)] = self.unconditional_embedding.repeat(uncond_count, 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

            if (self.use_pathgan or self.encode_embedding) and (self.time_embedder is not None or self.cond_embedder is not None):
                latent_embedder = getattr(self, f'e{res}')
                x = latent_embedder(x, torch.nn.functional.selu(emb))

        if for_features_matching:
            return x

        if self.use_pathgan:
            x = x = self.b4(x, img)
        else:
            cmap = None
            if self.c_dim > 0 and emb is not None:
                cmap = self.mapping(None, torch.nn.functional.selu(emb))
            x = self.b4(x, img, cmap)

        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


class WPlusEncoder(torch.nn.Module, CoditionalModule):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        latent_dim          = None,     # Latent size.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        w_dim               = 512,
        num_ws              = 18,
        block_type          = None,
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        **block_kwargs                  # Block arguments
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        block_class = DiscriminatorBlock
       
        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)

            if in_channels == 0:
                block = block_class(0, tmp_channels // 2, tmp_channels, resolution=res, without_downsample=True,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res * 2}', block)
                cur_layer_idx += block.num_layers

                block = block_class(tmp_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res}', block)
                cur_layer_idx += block.num_layers
            else:
                block = block_class(tmp_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res}', block)
                cur_layer_idx += block.num_layers

        self.block_resolutions.insert(0, img_resolution * 2)

        self.fc_ws = FullyConnectedLayer(channels_dict[4] * 4 * 4, w_dim * num_ws, activation='lrelu')

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = FullyConnectedLayer(channels_dict[4], latent_dim)

    def forward(self, img, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        pred_ws = self.fc_ws(x.view(x.size(0), -1))
        hidden_states = [x] + list(pred_ws.split(self.w_dim, dim=1))

        y = self.pool(x).view(x.size(0), -1)
        y = self.linear(y)

        return y, hidden_states

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


class ImageEncoder(torch.nn.Module, CoditionalModule):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        latent_dim          = None,     # Latent size.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        w_dim               = 512,
        block_type          = None,
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        **block_kwargs                  # Block arguments
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        block_class = DiscriminatorBlock
       
        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)

            if in_channels == 0:
                block = block_class(0, tmp_channels // 2, tmp_channels, resolution=res, without_downsample=True,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res * 2}', block)
                cur_layer_idx += block.num_layers

                block = block_class(tmp_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res}', block)
                cur_layer_idx += block.num_layers
            else:
                block = block_class(tmp_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
                setattr(self, f'b{res}', block)
                cur_layer_idx += block.num_layers

        self.block_resolutions.insert(0, img_resolution * 2)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = FullyConnectedLayer(channels_dict[4], latent_dim)

    def forward(self, img, update_emas=False, emb=None, **block_kwargs):
        _ = update_emas # unused
        x = None

        hidden_states = []
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')

            x, img = block(x, img, **block_kwargs)
            
            hidden_states.insert(0, x)

        y = self.pool(x).view(x.size(0), -1)
        y = self.linear(y)

        return y, hidden_states

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


class StyleGANv2AE(torch.nn.Module, CoditionalModule):
    def __init__(self,
        z_dim,                                  # Input latent (Z) dimensionality.
        c_dim,                                  # Conditioning label (C) dimensionality.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        img_resolution,                         # Output resolution.
        img_channels,                           # Number of output color channels.
        mapping_kwargs              = {},       # Arguments for MappingNetwork.
        sum_with_style              = False,
        disable_skip_connections    = False,
        time_embedder: torch.nn.Module = None, time_embedder_kwargs={}, cond_embedder: torch.nn.Module = None, cond_embedder_kwargs={},
        **synthesis_kwargs,                     # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.disable_skip_connections = disable_skip_connections

        if self.disable_skip_connections:
            print('WARNING: Skip connections are disabled!')

        if time_embedder is not None:
            self.time_embedder = time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None

        if cond_embedder is not None:
            self.cond_embedder = cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None
            cond_emb_dim = None

        if time_emb_dim is None and cond_emb_dim is not None:
            time_emb_dim = cond_emb_dim

        self.unconditional_embedding = None
        if time_emb_dim is not None:
            self.unconditional_embedding = torch.nn.Parameter(torch.randn(1, time_emb_dim))

        c_dim = time_emb_dim if time_emb_dim is not None else 0
        self.c_dim = c_dim

        self.sum_with_style = sum_with_style

        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        if synthesis_kwargs['block_type'] == 'style':
            self.image_encoder = WPlusEncoder(img_resolution, img_channels, z_dim, w_dim=w_dim, num_ws=self.num_ws, **synthesis_kwargs)
        else:
            self.image_encoder = ImageEncoder(img_resolution, img_channels, z_dim, **synthesis_kwargs)

    def forward(self, style_image, truncation_psi=1, truncation_cutoff=None, update_emas=False, t=None, condition=None, **synthesis_kwargs):
        emb = self.calculate_embedding(t, condition)
        if emb is None and (self.cond_embedder is not None or self.time_embedder is not None):
            emb = self.unconditional_embedding.repeat(style_image.size(0), 1)

        pred_z, encoded_style_img = self.image_encoder(style_image)

        if self.disable_skip_connections:
            encoded_style_img = None

        ws = self.mapping(pred_z, emb, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        img = self.synthesis(ws, update_emas=update_emas, style_features=encoded_style_img, **synthesis_kwargs)

        if self.sum_with_style:
            img = img + style_image
        return img
    
    def load_from_one_path_generator(self, weights: OrderedDict):
        cond_emb_weights = {key[len('cond_embedder.'):]: val for key, val in weights.items() if key.startswith('cond_embedder.')}
        if self.cond_embedder is not None and len(cond_emb_weights.keys()) > 0:
            self.cond_embedder.load_state_dict(cond_emb_weights)

        time_emb_weights = {key[len('time_embedder.'):]: val for key, val in weights.items() if key.startswith('time_embedder.')}
        if self.time_embedder is not None and len(time_emb_weights.keys()) > 0:
            self.time_embedder.load_state_dict(time_emb_weights)

        synthesis_weights = {key[len('synthesis.'):]: val for key, val in weights.items() if key.startswith('synthesis.')}
        self.synthesis.load_state_dict(synthesis_weights)

        mapping_weights = {key[len('mapping.'):]: val for key, val in weights.items() if key.startswith('mapping.')}
        self.mapping.load_state_dict(mapping_weights)
    
#----------------------------------------------------------------------------


if __name__ == "__main__":
    from diffusion_pipeline.models.embedders import CameraInfoMLP


    imgsz = 256
    emb_dim = 256
    latent_dim = 512

    # image_encoder = ImageEncoder(imgsz, 3, latent_dim=512)
    # hs = image_encoder(torch.rand(4, 3, imgsz, imgsz))

    # for x in hs:
    #     print("Enc: ", x.shape)

    # cam_h_dim = 16
    # cond_embedder = CameraInfoMLP
    # cond_embedder_kwargs = {
    #     'h_dim': cam_h_dim,
    #     'emb_dim': emb_dim
    # }

    g = StyleGANv2AE(
        z_dim=latent_dim,                      # Input latent (Z) dimensionality.
        c_dim=0,                      # Conditioning label (C) dimensionality.
        w_dim=512,                      # Intermediate latent (W) dimensionality.
        img_resolution=imgsz,             # Output resolution.
        img_channels=3,
        architecture='resnet',
        block_type='style',
        # cond_embedder=cond_embedder,
        # cond_embedder_kwargs=cond_embedder_kwargs
    )
    # weights = torch.load("/home/alexey/programming/science_work/diffusion_noise_generation/runs/generation_sidd/last.pt", weights_only=True)
    # g.load_from_one_path_generator(weights["state_dict"]["generator"])

    out = g(torch.rand(4, 3, imgsz, imgsz))