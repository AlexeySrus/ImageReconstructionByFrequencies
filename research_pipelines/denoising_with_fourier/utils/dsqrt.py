import torch


def diff_sqrt(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return torch.sqrt(x + eps) - torch.sqrt(torch.tensor(eps))
