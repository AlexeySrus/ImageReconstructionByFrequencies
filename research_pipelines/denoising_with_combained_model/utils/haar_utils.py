"""
Copyright 2021 Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import torch.nn as nn


class HaarForward(nn.Module):
    """
    Performs a 2d DWT Forward decomposition of an image using Haar Wavelets
    """
    alpha = 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Forward decomposition of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c * 4, h / 2, w / 2]
        """

        ll = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        lh = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hl = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hh = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        return torch.cat([ll,lh,hl,hh], axis=1)


class HaarInverse(nn.Module):
    """
    Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets
    """
    alpha = 1

    def _f(self, x, i, size):
        return x[:, size[1] * i : size[1] * (i + 1)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c / 4, h * 2, w * 2]
        """
        assert x.size(1) % 4 == 0, "The number of channels must be divisible by 4."
        size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
        out = torch.zeros(size, dtype=x.dtype, device=x.device)
        out[:,:,0::2,0::2] = self.alpha * (self._f(x, 0, size) + self._f(x, 1, size) + self._f(x, 2, size) + self._f(x, 3, size))
        out[:,:,0::2,1::2] = self.alpha * (self._f(x, 0, size) + self._f(x, 1, size) - self._f(x, 2, size) - self._f(x, 3, size))
        out[:,:,1::2,0::2] = self.alpha * (self._f(x, 0, size) - self._f(x, 1, size) + self._f(x, 2, size) - self._f(x, 3, size))
        out[:,:,1::2,1::2] = self.alpha * (self._f(x, 0, size) - self._f(x, 1, size) - self._f(x, 2, size) + self._f(x, 3, size))
        return out



class ConvHaarForward(nn.Module):
    """
    Performs a 2d DWT Forward decomposition of an image using Haar Wavelets implemented with Conv2d
    """
    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.w_conv = self.generate_dwt_kernel(n_channels)

    @staticmethod
    def generate_dwt_kernel(n_channels: int = 3):
        w = torch.zeros(n_channels * 4, n_channels, 2, 2)

        for k in range(n_channels):
            # A
            w[k + 4*k, k, :, :] = 1/4

            # # H
            # w[k + n_channels, k, 0, 0] = 1/4
            # w[k + n_channels, k, 0, 1] = -1/4
            # w[k + n_channels, k, 1, 0] = 1/4
            # w[k + n_channels, k, 1, 1] = -1/4

            # # V
            # w[k + n_channels * 2, k, 0, 0] = 1/4
            # w[k + n_channels * 2, k, 0, 1] = 1/4
            # w[k + n_channels * 2, k, 1, 0] = -1/4
            # w[k + n_channels * 2, k, 1, 1] = -1/4

            # # D
            # w[k + n_channels * 3, k, 0, 0] = 1/4
            # w[k + n_channels * 3, k, 0, 1] = -1/4
            # w[k + n_channels * 3, k, 1, 0] = -1/4
            # w[k + n_channels * 3, k, 1, 1] = 1/4

        a_param = torch.nn.Conv2d(n_channels, n_channels * 4, 2, 2)
        a_param.weight = torch.nn.Parameter(w, requires_grad=False)
        return a_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Forward decomposition of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c * 4, h / 2, w / 2]
        """

        return self.w_conv(x)


if __name__ == '__main__':
    dwt = HaarForward()
    cdwt = ConvHaarForward()

    t = torch.rand(1, 3, 512, 512)
    w1 = dwt(t)
    w2 = cdwt(t)

    print(torch.linalg.norm(w1[:, :3] - w2[:, :3]))
