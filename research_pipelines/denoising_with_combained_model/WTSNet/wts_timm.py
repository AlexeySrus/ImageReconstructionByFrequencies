from typing import Tuple, List, Optional
from collections import OrderedDict
import torch
import torch.nn as nn
from timm import create_model

from WTSNet.attention import CBAM, CoordinateAttention
from WTSNet.wtsnet import UpscaleByWaveletes, DownscaleByWaveletes, conv1x1, conv3x3, FeaturesProcessingWithLastConv
from utils.haar_utils import HaarForward, HaarInverse


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

from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

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
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
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
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
            padding_mode=padding_mode
        )
        self.norm = nn.LayerNorm(embed_dim)

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
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3, 3],
        sr_ratios=[8, 4, 2, 1, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=2, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )
        self.patch_embed5 = OverlapPatchEmbed(
            img_size=img_size // 32, patch_size=3, stride=2, in_chans=embed_dims[3], embed_dim=embed_dims[4]
        )

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        cur += depths[3]
        self.block5 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[4],
                    num_heads=num_heads[4],
                    mlp_ratio=mlp_ratios[4],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[4],
                )
                for i in range(depths[4])
            ]
        )
        self.norm5 = norm_layer(embed_dims[4])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[3]
        for i in range(self.depths[4]):
            self.block5[i].drop_path.drop_prob = dpr[cur + i]

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

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 5
        x, H, W = self.patch_embed5(x)
        for i, blk in enumerate(self.block5):
            x = blk(x, H, W)
        x = self.norm5(x)
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
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim, padding_mode=padding_mode)

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
        return self.forward_features(x)[: self._depth]

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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
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
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
        ),
    },
}


class MitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_cfg = mix_transformer_encoders['mit_b0']
        model_cfg['params']['depth'] = 5
        model_cfg['params']['patch_size'] = 4
        model_cfg['params']['out_channels'] = (3, 0, 8, 16, 32, 64)
        model_cfg['params']['img_size'] = 256
        self.encoder_model = MixVisionTransformerEncoder(**model_cfg['params'])

        self.hf_conv1 = conv3x3(32, 3*3)
        self.hf_conv2 = conv3x3(64, 3*3)
        self.hf_conv3 = conv1x1(160, 3*3)
        self.hf_conv4 = conv1x1(256, 3*3)
        self.hf_conv5 = conv1x1(512, 3*3)

        self.attn1 = CBAM(32, kernel_size=5)
        self.attn2 = CBAM(64, kernel_size=5)
        self.attn3 = CBAM(160, kernel_size=5)
        self.attn4 = CBAM(256, kernel_size=5)
        self.attn5 = CBAM(512, kernel_size=5)


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = (x - 0.5) *  2
        of1, of2, of3, of4, of5 = self.encoder_model(x)

        of1, _, sa1 = self.attn1(of1)
        of2, _, sa2 = self.attn2(of2)
        of3, _, sa3 = self.attn3(of3)
        of4, _, sa4 = self.attn4(of4)
        of5, _, sa5 = self.attn5(of5)

        hf1 = self.hf_conv1(of1)
        hf2 = self.hf_conv2(of2)
        hf3 = self.hf_conv3(of3)
        hf4 = self.hf_conv4(of4)
        hf5 = self.hf_conv5(of5)

        return hf1, hf2, hf3, hf4, hf5, [[sa1], [sa2], [sa3], [sa4], [sa5]]


class WTSNetSMP(nn.Module):
    def __init__(self, image_channels: int = 3):
        super().__init__()

        self.encoder_model = MitEncoder()

        self.iwt1 = UpscaleByWaveletes()
        self.iwt2 = UpscaleByWaveletes()
        self.iwt3 = UpscaleByWaveletes()
        self.iwt4 = UpscaleByWaveletes()


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pred_ll4 = nn.functional.interpolate(x, size=(x.size(2) // 16, x.size(3) // 16), mode='area')
        hf1, hf2, hf3, hf4, sa_list = self.encoder_model(x)

        pred_ll3 = self.iwt4(pred_ll4, hf4)
        pred_ll2 = self.iwt3(pred_ll3, hf3)
        pred_ll1 = self.iwt2(pred_ll2, hf2)
        pred_image = self.iwt1(pred_ll1, hf1)

        wavelets1 = torch.cat((pred_ll1, hf1), dim=1)
        wavelets2 = torch.cat((pred_ll2, hf2), dim=1)
        wavelets3 = torch.cat((pred_ll3, hf3), dim=1)
        wavelets4 = torch.cat((pred_ll4, hf4), dim=1)

        sa1, sa2, sa3, sa4 = sa_list
        sa1 = nn.functional.interpolate(sa1[0], (x.size(2), x.size(3)), mode='area')
        sa2 = nn.functional.interpolate(sa2[0], (x.size(2), x.size(3)), mode='area')
        sa3 = nn.functional.interpolate(sa3[0], (x.size(2), x.size(3)), mode='area')
        sa4 = nn.functional.interpolate(sa4[0], (x.size(2), x.size(3)), mode='area')

        return pred_image, [wavelets1, wavelets2, wavelets3, wavelets4], [sa1, sa2, sa3, sa4]


class TimmEncoderWithAttn(nn.Module):
    def __init__(self, model_name: str = 'resnet10t'):
        super().__init__()

        self.encoder_model = create_model(model_name, features_only=True)
        enc_channels = self.encoder_model.feature_info.channels()

        self.hf_conv1 = conv3x3(enc_channels[0], 3*3)
        self.hf_conv2 = conv3x3(enc_channels[1], 3*3)
        self.hf_conv3 = conv1x1(enc_channels[2], 3*3)
        self.hf_conv4 = conv1x1(enc_channels[3], 3*3)
        self.hf_conv5 = conv1x1(enc_channels[4], 3*3)

        self.attn1 = CBAM(enc_channels[0], kernel_size=5)
        self.attn2 = CBAM(enc_channels[1], kernel_size=5)
        self.attn3 = CBAM(enc_channels[2], kernel_size=5)
        self.attn4 = CBAM(enc_channels[3], kernel_size=5)
        self.attn5 = CBAM(enc_channels[4], kernel_size=5)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = (x - 0.5) * 2
        of1, of2, of3, of4, of5 = self.encoder_model(x)

        of1, _, sa1 = self.attn1(of1)
        of2, _, sa2 = self.attn2(of2)
        of3, _, sa3 = self.attn3(of3)
        of4, _, sa4 = self.attn4(of4)
        of5, _, sa5 = self.attn5(of5)

        hf1 = self.hf_conv1(of1)
        hf2 = self.hf_conv2(of2)
        hf3 = self.hf_conv3(of3)
        hf4 = self.hf_conv4(of4)
        hf5 = self.hf_conv5(of5)

        return hf1, hf2, hf3, hf4, hf5, [[sa1], [sa2], [sa3], [sa4], [sa5]]


class TimmEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet10t'):
        super().__init__()

        self.encoder_model = create_model(model_name, features_only=True)
        enc_channels = self.encoder_model.feature_info.channels()

        self.hf_conv1 = conv3x3(enc_channels[0], 3*3)
        self.hf_conv2 = conv3x3(enc_channels[1], 3*3)
        self.hf_conv3 = conv1x1(enc_channels[2], 3*3)
        self.hf_conv4 = conv1x1(enc_channels[3], 3*3)
        self.hf_conv5 = conv1x1(enc_channels[4], 3*3)

    def o2a(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.clamp(inp.mean(dim=1).unsqueeze(1) + 0.5, 0, 1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x = (x - 0.5) * 2
        of1, of2, of3, of4, of5 = self.encoder_model(x)

        sa1 = self.o2a(of1)
        sa2 = self.o2a(of2)
        sa3 = self.o2a(of3)
        sa4 = self.o2a(of4)
        sa5 = self.o2a(of5)

        hf1 = self.hf_conv1(of1)
        hf2 = self.hf_conv2(of2)
        hf3 = self.hf_conv3(of3)
        hf4 = self.hf_conv4(of4)
        hf5 = self.hf_conv5(of5)

        return hf1, hf2, hf3, hf4, hf5, [[sa1], [sa2], [sa3], [sa4], [sa5]]



class WTSNetTimm(nn.Module):
    def __init__(self, model_name: str = 'resnet10t', image_channels: int = 3, use_clipping: bool = False):
        super().__init__()

        if model_name == 'mit':
            print('Use mit model')
            self.encoder_model = MitEncoder()
        else:
            print('Use timm model')
            self.encoder_model = TimmEncoderWithAttn(model_name=model_name)

        self.iwt1 = UpscaleByWaveletes()
        self.iwt2 = UpscaleByWaveletes()
        self.iwt3 = UpscaleByWaveletes()
        self.iwt4 = UpscaleByWaveletes()
        self.iwt5 = UpscaleByWaveletes()

        self.use_clipping = use_clipping


    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        pred_ll5 = nn.functional.interpolate(x, size=(x.size(2) // 32, x.size(3) // 32), mode='area')
        hf1, hf2, hf3, hf4, hf5, sa_list = self.encoder_model(x)

        if self.use_clipping:
            hf1 = torch.clamp(hf1, -0.5, 0.5)
            hf2 = torch.clamp(hf2, -0.5, 0.5)
            hf3 = torch.clamp(hf3, -0.5, 0.5)
            hf4 = torch.clamp(hf4, -0.5, 0.5)
            hf5 = torch.clamp(hf5, -0.5, 0.5)

        pred_ll4 = self.iwt5(pred_ll5, hf5)
        pred_ll3 = self.iwt4(pred_ll4, hf4)
        pred_ll2 = self.iwt3(pred_ll3, hf3)
        pred_ll1 = self.iwt2(pred_ll2, hf2)
        pred_image = self.iwt1(pred_ll1, hf1)

        wavelets1 = torch.cat((pred_ll1, hf1), dim=1)
        wavelets2 = torch.cat((pred_ll2, hf2), dim=1)
        wavelets3 = torch.cat((pred_ll3, hf3), dim=1)
        wavelets4 = torch.cat((pred_ll4, hf4), dim=1)
        wavelets5 = torch.cat((pred_ll5, hf5), dim=1)

        sa1, sa2, sa3, sa4, sa5 = sa_list
        sa1 = nn.functional.interpolate(sa1[0], (x.size(2), x.size(3)), mode='area')
        sa2 = nn.functional.interpolate(sa2[0], (x.size(2), x.size(3)), mode='area')
        sa3 = nn.functional.interpolate(sa3[0], (x.size(2), x.size(3)), mode='area')
        sa4 = nn.functional.interpolate(sa4[0], (x.size(2), x.size(3)), mode='area')
        sa5 = nn.functional.interpolate(sa5[0], (x.size(2), x.size(3)), mode='area')

        return pred_image, [wavelets1, wavelets2, wavelets3, wavelets4, wavelets5], [sa1, sa2, sa3, sa4, sa5]


if __name__ == '__main__':
    import cv2
    import numpy as np
    from fvcore.nn import FlopCountAnalysis
    from timeit import default_timer

    device = 'cpu'

    model = WTSNetTimm(model_name='efficientnet_b0').to(device)
    model.eval()

    wsize = 256

    t = torch.rand(5, 3, wsize, wsize)

    with torch.no_grad():
        _ = model(t)

    N = 100
    start_time = default_timer()
    with torch.no_grad():
        for _ in range(N):
            a = model(t)
    finish_time = default_timer()

    print('Inference time: {:.2f}'.format((finish_time - start_time) / N))

    img_path = '/media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/cl_img7.jpeg'
    
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    img = np.random.randint(0, 256, size=(wsize, wsize, 3), dtype=np.uint8)
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp)

    print('Rand MSE: {}'.format(torch.nn.functional.mse_loss(out[0], inp).item()))

    for block_output, block_name in zip(out[1], ('d{}'.format(i) for i in range(1, 6 + 1))):
        print('Shape of {}: {}'.format(block_name, block_output.shape))

    # print()

    # for block_output, block_name in zip(out[2], ('sa_{}'.format(i) for i in range(1enc_channels = self., 6 + 1))):
    #     print('Shape of {}: {}'.format(block_name, block_output.shape))

    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Params: {} params'.format(params))

    # print('TFOPS: {:.1f}'.format(FlopCountAnalysis(model, inp).total() / 1E+9))

    # model_cfg = mix_transformer_encoders['mit_b0']
    # model_cfg['params']['depth'] = 5
    # model_cfg['params']['patch_size'] = 4
    # model_cfg['params']['out_channels'] = (3, 0, 8, 16, 32, 64)
    # model = MixVisionTransformerEncoder(**model_cfg['params'])
    # model = create_model('resnet10t', features_only=True)
    # print(model.feature_info.channels())
    # out = model(inp)
    # for o in out:
    #     print(o.shape)
    dwt = DownscaleByWaveletes()
    iwt = UpscaleByWaveletes()

    x = torch.clone(inp)
    pred_ll5 = nn.functional.interpolate(x, size=(x.size(2) // 32, x.size(3) // 32), mode='area')

    start_time = default_timer()
    for _ in range(100):
        ll1, hf1 = dwt(x)
        ll2, hf2 = dwt(ll1)
        ll3, hf3 = dwt(ll2)
        ll4, hf4 = dwt(ll3)
        ll5, hf5 = dwt(ll4)

        pred_ll4 = iwt(pred_ll5, hf5)
        pred_ll3 = iwt(pred_ll4, hf4)
        pred_ll2 = iwt(pred_ll3, hf3)
        pred_ll1 = iwt(pred_ll2, hf2)
        pred_image = iwt(pred_ll1, hf1)
    finish_time = default_timer()

    print('Wavelets full path time: {:.2f}'.format((finish_time - start_time) / 100))

    print(hf1.min(), hf1.max())

    print('L2 between area and wavelets: {}'.format(torch.linalg.norm(pred_ll5 - ll5)))
    print(pred_ll5[0, :, 1, 1], ll5[0, :, 1, 1])

    print(pred_image.min(), pred_image.max())

    print('Wavelets MSE: {}'.format(torch.linalg.norm(pred_image - inp)))

    x = torch.LongTensor(
        [
            [1, 2, 3, 4, 5, 1],
            [6, 7, 8, 9, 10, 1],
            [11, 12, 13, 14, 15, 1],
            [16, 17, 18, 19, 20, 1],
            [21, 22, 23, 24, 25, 1],
            [26, 27, 28, 29, 30, 31]
        ]
    ).unsqueeze(0).unsqueeze(0)
    print(x.shape)

    ll1, hf1 = dwt(x)
    ll1 = ll1.to(torch.long)[0][0]
    
    pl1 = nn.functional.interpolate(x.to(torch.float32), size=(x.size(2) // 2, x.size(3) // 2), mode='area')
    pl1 = pl1.to(torch.long)[0][0]

    print(ll1)
    print(pl1)