""" 
PyTorch implementation of CBAM: Convolutional Block Attention Module

As described in https://arxiv.org/pdf/1807.06521

The attention mechanism is achieved by using two different types of attention gates: 
channel-wise attention and spatial attention. The channel-wise attention gate is applied 
to each channel of the input feature map, and it allows the network to focus on the most 
important channels based on their spatial relationships. The spatial attention gate is applied 
to the entire input feature map, and it allows the network to focus on the most important regions 
of the image based on their channel relationships.
"""


import torch
from torch import nn


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class FullComplexSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(FullComplexSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, dtype=torch.cfloat, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        avg_out = torch.mean(z, dim=1, keepdim=True)
        _, max_indices = torch.max(torch.abs(z), dim=1, keepdim=True)
        max_out = retrieve_elements_from_indices(z, max_indices)
        
        # x_max, _ = torch.max(x, dim=1, keepdim=True)
        # max_out = torch.fft.fft2(x_max, norm='ortho')
        # max_out = torch.fft.fftshift(max_out)
        
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return z * attn, attn


class ComplexSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ComplexSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='zeros')
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = torch.abs(z)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return z * attn.to(torch.cfloat), attn


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False, padding_mode='reflect')
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.sigmoid(out)
        return x * attn, attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attn = self.sigmoid(out)
        return x * attn, attn


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x, ca_tensor = self.ca(x)
        x, sa_tensor = self.sa(x)
        return x, ca_tensor, sa_tensor


class ComplexSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(ComplexSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.key = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.value = nn.Linear(input_dim, input_dim, dtype=torch.cfloat)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(torch.abs(scores)).to(torch.cfloat)
        weighted = torch.bmm(attention, values)
        return weighted


def real_imaginary_relu(z):
    return nn.functional.leaky_relu(z.real) + 1.j * nn.functional.leaky_relu(z.imag)


class WindowBasedSelfAttention(nn.Module):
    def __init__(self, window_size:int = 64):
        super(WindowBasedSelfAttention, self).__init__()
        self.self_attention = ComplexSelfAttention(window_size ** 2 // 4)

        self.down = nn.Linear(window_size ** 2, window_size ** 2 // 4, dtype=torch.cfloat)
        self.up = nn.Linear(window_size ** 2 // 4, window_size ** 2, dtype=torch.cfloat)

        self.wsize = window_size

    def forward(self, x):
        features_folds = x.unfold(
            2, size=self.wsize, step=self.wsize // 2).unfold(
                3, size=self.wsize, step=self.wsize // 2).contiguous()

        four_folds = torch.fft.fft2(features_folds)
        four_folds = torch.fft.fftshift(four_folds)

        # _, max_indices = torch.max(torch.abs(four_folds), dim=1, keepdim=True)
        # folds = retrieve_elements_from_indices(four_folds, max_indices)
        four_folds = torch.mean(four_folds, dim=1, keepdim=True)

        init_folds_shape = folds.shape        

        folds = folds.view(
            folds.size(0), 
            folds.size(1) * folds.size(2) * folds.size(3),
            folds.size(4) * folds.size(5)
        )

        dfolds = self.down(folds)
        dfolds = real_imaginary_relu(dfolds)
        out = self.self_attention(dfolds)
        out = self.up(out)

        out = out.view(*init_folds_shape)

        loc_attn = torch.sigmoid(out)

        out = four_folds * loc_attn

        out = torch.fft.ifftshift(out)
        out = torch.fft.ifft2(out).real

        corners = out[:, :, ::2, ::2]
        anchors = out[:, :, 1::2, 1::2]
        path_size = self.wsize

        corners[:, :, :-1, :-1, path_size//2:, path_size//2:] = \
                                        corners[:, :, :-1, :-1, path_size//2:, path_size//2:] / 2 + \
                                        anchors[:, :, :, :, :path_size//2, :path_size//2] / 2

        corners[:, :, :-1, 1:, path_size//2:, :path_size//2] = \
                                                corners[:, :, :-1, 1:, path_size//2:, :path_size//2] / 2 + \
                                                anchors[:, :, :, :, :path_size//2, path_size//2:] / 2

        corners[:, :, 1:, :-1, :path_size//2, path_size//2:] = \
                                                corners[:, :, 1:, :-1, :path_size//2, path_size//2:] / 2 + \
                                                anchors[:, :, :, :, path_size//2:, :path_size//2] / 2

        corners[:, :, 1:, 1:, :path_size//2, :path_size//2] = \
                                                corners[:, :, 1:, 1:, :path_size//2, :path_size//2] / 2 + \
                                                anchors[:, :, :, :, path_size//2:, path_size//2:] / 2


        out = corners
        out = torch.cat([out[:, :, :, i] for i in range(out.size(3))], dim=4)
        out = torch.cat([out[:, :, i] for i in range(out.size(2))], dim=2)

        out = torch.cat([out[:, :, :, i] for i in range(out.size(3))], dim=4)
        out = torch.cat([out[:, :, i] for i in range(out.size(2))], dim=2)

        return out