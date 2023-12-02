import torch
from utils.haar_utils import HaarForward


class SparceLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module = torch.nn.functional.smooth_l1_loss, threshold: float = 1E-3):
        super().__init__()
        self.threshold = threshold
        self.base_loss = base_loss

    def forward(self, pred: torch.Tensor, trurh: torch.Tensor) -> torch.Tensor:
        difference = torch.abs(pred - trurh)

        significant_indexes = difference > self.threshold

        origin_loss = self.base_loss(pred, trurh)

        if significant_indexes.sum() == 0:
            return origin_loss

        specifying_loss = difference[significant_indexes].sum() / significant_indexes.sum()

        return origin_loss + specifying_loss


if __name__ == '__main__':
    import cv2
    import numpy as np
    device = 'cpu'

    img_path = '/media/alexey/SSDData/datasets/denoising_dataset/base_clear_images/cl_img7.jpeg'
    
    wsize = 256
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:wsize, :wsize, ::-1]
    img = np.random.randint(0, 256, size=(wsize, wsize, 3), dtype=np.uint8)
    inp = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    inp = inp.to(device)

    ahvd = HaarForward()(inp)
    hvd = ahvd[:, 3:]

    sl1 = torch.nn.SmoothL1Loss()
    sparce_l1 = SparceLoss()

    hvd2 = torch.clone(hvd)
    hvd2[:, :, 65, 65] += 0.3
    hvd2.requires_grad = True

    optim = torch.optim.RAdam([hvd2], lr=0.001)

    for i in range(50000):
        optim.zero_grad()
        loss = sparce_l1(hvd2, hvd)
        loss.backward()
        optim.step()

        if (i + 1) % 100 == 0:
            print(loss.item())
