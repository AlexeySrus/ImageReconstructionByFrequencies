from typing import Callable
import torch
from torch import nn
import numpy as np
import math

from utils.filters import get_log_kernel, load_filter


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                    y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel


def create_butterworth_high_pass_filter(width, height, d, n):
    hp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, hp_filter.shape[1]):  # image width
        for j in range(0, hp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            hp_filter[j, i] = 1 / (1 + math.pow((d / radius), (2 * n)))
    return hp_filter


# create a butterworth low pass filter

def create_butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width, 2), np.float32)
    centre = (width / 2, height / 2)

    for i in range(0, lp_filter.shape[1]):  # image width
        for j in range(0, lp_filter.shape[0]):  # image height
            radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
            lp_filter[j, i] = 1 / (1 + math.pow((radius / d), (2 * n)))
    return lp_filter


class HightFrequencyImageComponent(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()

        # kernel = 1.0 - generate_batt(shape, 500, 1).astype(np.float32)
        kernel = create_butterworth_high_pass_filter(shape[0], shape[1], 15, 2)[..., 0]
        self.image_shape = shape

        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(torch.cfloat)
        
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def apply_fft_kernel(self, x):
        return x*self.kernel

    def forward(self, z):
        n_fourier_transform_x = self.apply_fft_kernel(
            torch.fft.fftshift(z)
        )
        return torch.fft.ifftshift(n_fourier_transform_x)


class HightFrequencyFFTLoss(nn.Module):
    def __init__(self, shape: tuple, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'sum'], 'Not supported reduction method: {}'.format(reduction)
        self.reduction = reduction

        hight_pass_kernel = create_butterworth_high_pass_filter(shape[0], shape[1], 15, 2)[..., 0]

        hight_pass_kernel = torch.from_numpy(hight_pass_kernel).unsqueeze(0).unsqueeze(0)
        hight_pass_kernel = torch.fft.fftshift(hight_pass_kernel)
        hight_pass_kernel = hight_pass_kernel[:, :, :, :shape[1] // 2 + 1]
        
        self.kernel = nn.Parameter(hight_pass_kernel, requires_grad=False)
        self.kernel_sum = nn.Parameter(hight_pass_kernel.sum((1, 2, 3)), requires_grad=False)

    def calculate_err_with_mask(self, base_err: torch.Tensor, warp_func: Callable[[torch.Tensor], torch.Tensor]):
        err = warp_func(base_err)

        err = (err * self.kernel).sum((1, 2, 3))
        err = err / (self.kernel_sum + 1e-6)

        return err

    def forward(self, x_pred, x_truth):
        z_pred = torch.fft.rfft2(x_pred, norm='forward')
        z_truth = torch.fft.rfft2(x_truth, norm='forward')

        base_err = z_pred - z_truth

        abs_err = self.calculate_err_with_mask(base_err, torch.abs)
        phase_err = self.calculate_err_with_mask(base_err, lambda v: torch.abs(torch.angle(v)))

        err = abs_err / 2 + phase_err / 2

        if self.reduction == 'mean':
            err = err.mean()
        else:
            err = err.sum()

        return err
    

class FrequencyRelationLoss(nn.Module):
    def __init__(self, shape: tuple, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['mean', 'sum'], 'Not supported reduction method: {}'.format(reduction)
        self.reduction = reduction

        hight_pass_kernel = create_butterworth_high_pass_filter(shape[0], shape[1], 15, 2)[..., 0]

        hight_pass_kernel = torch.from_numpy(hight_pass_kernel).unsqueeze(0).unsqueeze(0)
        hight_pass_kernel = torch.fft.fftshift(hight_pass_kernel)
        hight_pass_kernel = hight_pass_kernel[:, :, :, :shape[1] // 2 + 1]
        
        self.hp_kernel = nn.Parameter(hight_pass_kernel, requires_grad=False)

    def forward(self, x_pred):
        z_pred = torch.fft.rfft2(x_pred, norm='ortho')

        h_x = torch.fft.irfft2(z_pred * self.hp_kernel, norm='ortho')

        err = torch.log(torch.abs(h_x)).mean((1, 2, 3))
        err = 1.0 - err

        if self.reduction == 'mean':
            err = err.mean()
        else:
            err = err.sum()

        return err


class HFENLoss(nn.Module): # Edge loss with pre_smooth
    """Calculates high frequency error norm (HFEN) between target and
     prediction used to quantify the quality of reconstruction of edges
     and fine features.

     Uses a rotationally symmetric LoG (Laplacian of Gaussian) filter to
     capture edges. The original filter kernel is of size 15×15 pixels,
     and has a standard deviation of 1.5 pixels.
     ks = 2 * int(truncate * sigma + 0.5) + 1, so use truncate=4.5

     HFEN is computed as the norm of the result obtained by LoG filtering the
     difference between the reconstructed and reference images.

    [1]: Ravishankar and Bresler: MR Image Reconstruction From Highly
    Undersampled k-Space Data by Dictionary Learning, 2011
        https://ieeexplore.ieee.org/document/5617283
    [2]: Han et al: Image Reconstruction Using Analysis Model Prior, 2016
        https://www.hindawi.com/journals/cmmm/2016/7571934/

    Parameters
    ----------
    img1 : torch.Tensor or torch.autograd.Variable
        Predicted image
    img2 : torch.Tensor or torch.autograd.Variable
        Target image
    norm: if true, follows [2], who define a normalized version of HFEN.
        If using RelativeL1 criterion, it's already normalized.
    """
    def __init__(self, loss_f=None, kernel='log', kernel_size=15, sigma = 2.5, norm = False): #1.4 ~ 1.5
        super(HFENLoss, self).__init__()
        # can use different criteria
        self.criterion = loss_f
        self.norm = norm
        #can use different kernels like DoG instead:
        kernel = get_log_kernel(kernel_size, sigma)
        self.filter = load_filter(kernel=kernel, kernel_size=kernel_size)

    def forward(self, img1, img2):
        self.filter.to(img1.device)
        # HFEN loss
        log1 = self.filter(img1)
        log2 = self.filter(img2)
        hfen_loss = self.criterion(log1, log2)
        if self.norm:
            hfen_loss /= img2.norm()
        return hfen_loss


if __name__ == '__main__':
    loss = HightFrequencyFFTLoss(shape=(512, 512))
    t1 = torch.rand(1, 3, 512, 512)
    t2 = torch.rand(1, 3, 512, 512)
    lv = loss(t1, t2)
    print(lv)
