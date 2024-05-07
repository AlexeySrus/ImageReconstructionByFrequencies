from typing import List, Optional

import kornia
import torch


def iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    inputs = pred.view(-1)
    targets = target.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


class EdgeLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_loss = lambda x, y: torch.nn.functional.l1_loss(x, y)

    def __call__(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        total_loss = sum(
            [
                self.base_loss(
                    kornia.filters.laplacian(pred, kernel_size=ks),
                    kornia.filters.laplacian(truth, kernel_size=ks)
                )
                for ks in [3, 5, 7]
            ]
        )

        return total_loss / 3.0
