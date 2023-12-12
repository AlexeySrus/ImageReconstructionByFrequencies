import torch
from torch import nn
import numpy as np

from utils.filters import get_log_kernel, load_filter


def generate_batt(size=(5, 5), d0=5, n=2):
    kernel = np.fromfunction(
        lambda x, y: \
            1 / (1 + (((x - size[0] // 2) ** 2 + (
                    y - size[1] // 2) ** 2) ** 1 / 2) / d0) ** n,
        (size[0], size[1])
    )
    return kernel


class HightFrequencyImageComponent(nn.Module):
    def __init__(self, shape: tuple):
        super().__init__()

        kernel = 1.0 - generate_batt(shape, 500, 1).astype(np.float32)
        self.image_shape = shape

        kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.to(torch.cfloat)
        
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def apply_fft_kernel(self, x):
        return x*self.kernel[:, -x.size(2):]

    def forward(self, z):
        n_fourier_transform_x = self.apply_fft_kernel(
            torch.fft.fftshift(z)
        )
        return torch.fft.ifftshift(n_fourier_transform_x)


class HightFrequencyFFTLoss(nn.Module):
    def __init__(self, shape: tuple, base_loss: nn.Module = torch.nn.functional.l1_loss):
        super().__init__()
        self.hf_exrtractor = HightFrequencyImageComponent(shape)
        self.base_loss_fn = base_loss

    def forward(self, z_pred, z_truth):
        hf_z_pred = self.hf_exrtractor(z_pred)
        hf_z_truth = self.hf_exrtractor(z_truth)
        return self.base_loss_fn(hf_z_pred, hf_z_truth)


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
    loss = HFENLoss(torch.nn.functional.smooth_l1_loss)
    t1 = torch.rand(1, 3, 512, 512)
    t2 = torch.rand(1, 3, 512, 512)
    lv = loss(t1, t2)
    print(lv)
