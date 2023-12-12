import torch
from torch import nn
import numpy as np

from utils.filters import get_log_kernel, load_filter


def denorm(x, min_max=(-1.0, 1.0)):
    '''
        Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    '''
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")


def norm(x):
    #Normalize (z-norm) from [0,1] range to [-1,1]
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")


class FFTloss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss, reduction='mean'):
        super(FFTloss, self).__init__()
        self.criterion = loss_f(reduction=reduction)

    def forward(self, img1, img2):
        zeros=torch.zeros(img1.size()).to(img1.device)
        return self.criterion(torch.fft(torch.stack((img1,zeros),-1),2),torch.fft(torch.stack((img2,zeros),-1),2))


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
