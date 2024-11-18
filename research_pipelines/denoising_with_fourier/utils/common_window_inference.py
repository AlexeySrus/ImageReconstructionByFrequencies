from typing import Union, List, Any, Callable
import numpy as np
import torch
from tqdm import tqdm


def denoise_inference(
        tensor_img: torch.Tensor,
        inference_function: Callable[[torch.Tensor], torch.Tensor],
        window_size: int = 224,
        batch_size: int = 32,
        verbose: bool = False,
        crop_size: int = 0) -> torch.Tensor:
    crop_d = crop_size
    output_size = window_size - crop_d * 2

    d = (window_size - output_size) // 2

    margin_width = (
        output_size - tensor_img.size(2) % output_size
    ) * (tensor_img.size(2) % output_size != 0)

    margin_height = (
        output_size - tensor_img.size(1) % output_size
    ) * (tensor_img.size(1) % output_size != 0)

    padded_tensor = torch.nn.functional.pad(
        tensor_img.unsqueeze(0),
        [d, d + margin_width, d, d + margin_height],
        mode='reflect'
    ).squeeze(dim=0)

    crops = []
    for i in range(padded_tensor.size(1) // output_size):
        for j in range(padded_tensor.size(2) // output_size):
            crops.append(
                padded_tensor[
                    :,
                    i*output_size:i*output_size + window_size,
                    j*output_size:j*output_size + window_size
                ]
            )

    crops = torch.split(torch.stack(crops), batch_size)
    crops_buffer = tqdm(crops) if verbose else crops

    outs = torch.cat(
        [
            inference_function(bc)[:, :, crop_d:-crop_d, crop_d:-crop_d] \
                if crop_d > 0 else \
                    inference_function(bc)
            for bc in crops_buffer
        ],
        dim=0
    )

    result_transform_image = outs.view(
        padded_tensor.size(1) // output_size,
        padded_tensor.size(2) // output_size, 
        tensor_img.size(0),
        output_size,
        output_size
    )

    result_image = torch.cat(
        tuple(torch.cat(tuple(result_transform_image), dim=2)),
        dim=2
    )

    result_img = result_image[
                    :,
                    :tensor_img.size(1),
                    :tensor_img.size(2)
                ]
    return result_img
