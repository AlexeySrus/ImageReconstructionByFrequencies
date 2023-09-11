import numpy as np
import torch
from tqdm import tqdm
from .tensor_utils import TensorRotate, rotate_tensor, preprocess_image


def denoise_inference(
        tensor_img: torch.Tensor,
        model: torch.nn.Module,
        iwt: torch.nn.Module,
        window_size: int = 224,
        batch_size: int = 32,
        verbose: bool = False) -> torch.Tensor:
    # output_size = model(torch.zeros(1, 3, window_size, window_size).to(device))[0][0].size(2) * 2
    crop_d = window_size // 32 // 2
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
    ).squeeze()

    predicted_images = []

    transforms = [
        TensorRotate.NONE,
        TensorRotate.ROTATE_90_CLOCKWISE,
        TensorRotate.ROTATE_180,
        TensorRotate.ROTATE_90_COUNTERCLOCKWISE
    ]

    for transform in transforms:
        transform_padded_image = rotate_tensor(padded_tensor, transform)

        crops = []
        for i in range(transform_padded_image.size(1) // output_size):
            for j in range(transform_padded_image.size(2) // output_size):
                crops.append(
                    transform_padded_image[
                        :,
                        i*output_size:i*output_size + window_size,
                        j*output_size:j*output_size + window_size
                    ]
                )

        crops = torch.split(torch.stack(crops), batch_size)
        crops_buffer = tqdm(crops) if verbose else crops

        outs = torch.cat(
            [
                iwt(*torch.split(model(bc)[0][0], 3, dim=1))[:, :, crop_d:-crop_d, crop_d:-crop_d]
                for bc in crops_buffer
            ],
            dim=0
        )

        result_transform_image = outs.view(
            transform_padded_image.size(1) // output_size,
            transform_padded_image.size(2) // output_size, 3,
            output_size,
            output_size
        )
        result_transform_image = torch.cat(
            tuple(torch.cat(tuple(result_transform_image), dim=2)),
            dim=2
        )

        predicted_images.append(result_transform_image)

    predicted_images = torch.stack(
        [
            rotate_tensor(predicted_images[i], transform)
            for i, transform in enumerate(transforms[:1] + transforms[1:][::-1])
        ],
        dim=0
    )
    result_image = torch.mean(predicted_images, dim=0)

    result_img = result_image[
                    :,
                    :tensor_img.size(1),
                    :tensor_img.size(2)
                ]
    return result_img
