from typing import Tuple, List, Optional, Union
from collections import OrderedDict
import torch
import torch.nn as nn


padding_mode: str = 'reflect'


# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math
import torch
import torch.nn as nn
from functools import partial

from timm.layers import to_2tuple, trunc_normal_


def drop_complex_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor.to(torch.cfloat)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_complex_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class RISwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()

    def forward(self, z):
        return nn.functional.silu(z.real) + 1.j * nn.functional.silu(z.imag)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], *,
                    eps: float = 1e-5,  elementwise_affine: bool = True, dtype = torch.cfloat):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=RISwish, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, dtype=torch.cfloat)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias, dtype=torch.cfloat)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias, dtype=torch.cfloat)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, dtype=torch.cfloat)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, dtype=torch.cfloat)
            self.norm = LayerNorm(dim, dtype=torch.cfloat)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.abs(attn).softmax(dim=-1).to(torch.cfloat)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=RISwish,
        norm_layer=LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, dtype=torch.cfloat)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, dtype=torch.cfloat)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
            padding_mode=padding_mode,
            dtype=torch.cfloat
        )
        self.norm = LayerNorm(embed_dim, dtype=torch.cfloat)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    _depth = 5

    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512, 512],
        num_heads=[1, 2, 4, 8, 4],
        mlp_ratios=[4, 4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=LayerNorm,
        depths=[3, 4, 6, 3, 3],
        sr_ratios=[8, 4, 2, 1, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.ModuleList(
            [
                OverlapPatchEmbed(
                    img_size=img_size // (2 ** (0 if d == 0 else d + 1)), 
                    patch_size=(7 if d == 0 else 3), 
                    stride=2, 
                    in_chans=in_chans if d == 0 else embed_dims[d-1], 
                    embed_dim=embed_dims[d]
                )
                for d in range(self._depth)
            ]
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        zero_first_depts = [0] + depths

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Block(
                            dim=embed_dims[d],
                            num_heads=num_heads[d],
                            mlp_ratio=mlp_ratios[d],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(zero_first_depts[:d+1]) + i],
                            norm_layer=norm_layer,
                            sr_ratio=sr_ratios[d],
                        )
                        for i in range(depths[d])
                    ]
                )
                for d in range(self._depth)
            ]
        )
        self.norms = nn.ModuleList(
            [
                norm_layer(embed_dims[d])
                for d in range(self._depth)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0

        for d in range(self._depth):
            for i in range(self.depths[d]):
                self.block1[i].drop_path.drop_prob = dpr[cur + i]

            cur += self.depths[d]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for d in range(self._depth):
            x, H, W = self.patch_embeds[d](x)
            for i, blk in enumerate(self.blocks[d]):
                x = blk(x, H, W)
            x = self.norms[d](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim, padding_mode=padding_mode, dtype=torch.cfloat)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# ---------------------------------------------------------------
# End of NVIDIA code
# ---------------------------------------------------------------

from segmentation_models_pytorch.encoders._base import EncoderMixin  # noqa E402


class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

    def make_dilated(self, *args, **kwargs):
        raise ValueError("MixVisionTransformer encoder does not support dilated mode")

    def set_in_channels(self, in_channels, *args, **kwargs):
        if in_channels != 3:
            raise ValueError("MixVisionTransformer encoder does not support in_channels setting other than 3")

    def forward(self, x):
        return self.forward_features(x)

    def load_state_dict(self, state_dict):
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict)


def get_pretrained_cfg(name):
    return {
        "url": "https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/{}.pth".format(name),
        "input_space": "RGB",
        "input_size": [3, 224, 224],
        "input_range": [0, 1],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


mix_transformer_encoders = {
    "mit_b0": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b0"),
        },
        "params": dict(
            out_channels=(3, 0, 32, 64, 160, 256, 512),
            patch_size=4,
            embed_dims=[32, 64, 160, 256, 512],
            num_heads=[1, 2, 5, 8, 8],
            mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1, 1],
            drop_rate=0.0,
            drop_path_rate=0.0,
        ),
    },
    "mit_b1": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b1"),
        },
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b2": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b2"),
        },
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b3": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b3"),
        },
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b4": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b4"),
        },
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
    "mit_b5": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": get_pretrained_cfg("mit_b5"),
        },
        "params": dict(
            out_channels=(3, 0, 64, 128, 320, 512),
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
}


class MitEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, depth: int = 5):
        super().__init__()
        assert depth >= 1, 'Need set \'depth\' >= 1, recieved: {}'.format(depth)
        self.depth = depth
        model_cfg = mix_transformer_encoders['mit_b0']
        model_cfg['params']['depth'] = 5
        model_cfg['params']['patch_size'] = 4
        model_cfg['params']['out_channels'] = (3, 0, 8, 16, 32, 64)
        model_cfg['params']['img_size'] = 256
        model_cfg['params']['depth'] = depth
        model_cfg['params']['in_chans'] = in_ch
        self.encoder_model = MixVisionTransformerEncoder(**model_cfg['params'])
        self.encoder_features = (32, 64, 160, 256, 512)[:depth]

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self.encoder_model(x)


if __name__ == '__main__':
    model = MitEncoder(in_ch=16, depth=2)
    t = torch.zeros(1, 16, 256, 256, dtype=torch.cfloat)
    out = model(t)
    for o in out:
        print(o.shape)
