from typing import List
import os
import sys
from pathlib import Path

from enum import Enum
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageOps
import pydicom as dicom
import io
import onnxruntime as ort

import streamlit as st
from stqdm import stqdm
from streamlit_image_comparison import image_comparison

DIR_PATH = os.path.dirname(__file__)
ONNX_MODEL_PATH: str = os.path.join(DIR_PATH, 'mri_tfaunet.onnx')
IMAGE_SIZE: int = 256
USE_UFORMER: bool = True


def run_onnx_model_on_batch(
        model: ort.capi.onnxruntime_inference_collection.InferenceSession,
        imgs_batch: np.ndarray) -> np.ndarray:
    net_out = model.run(
        ['output'],
        {'input': imgs_batch}
    )
    net_out = net_out[0]
    return net_out


class TensorRotate(Enum):
    """Rotate enumerates class"""
    NONE = lambda x: x
    HORISONTAL_FLIP = lambda x: x.flip(2)
    ROTATE_90_CLOCKWISE = lambda x: x.transpose(1, 2).flip(2)
    ROTATE_180 = lambda x: x.flip(1, 2)
    ROTATE_90_COUNTERCLOCKWISE = lambda x: x.transpose(1, 2).flip(1)


def rotate_tensor(img: np.ndarray, rot_value: TensorRotate) -> np.ndarray:
    """Rotate image tensor

    Args:
        img: tensor in CHW format
        rot_value: element of TensorRotate class, possible values
            TensorRotate.NONE,
            TensorRotate.HORISONTAL_FLIP,
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_180,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE,

    Returns:
        Rotated image in same of input format
    """
    return rot_value(img)


def array_split(arr: np.ndarray, batch_size: int) -> List[np.ndarray]:
    if arr.shape[0] <= batch_size:
        return [arr]
    
    last_base_elem = batch_size * (arr.shape[0] // batch_size)
    split_arr = np.split(arr[:last_base_elem], arr.shape[0] // batch_size)
    if arr.shape[0] % batch_size != 0:
        split_arr += [arr[last_base_elem:]]

    return split_arr


def window_denoise_inference(
        model: ort.capi.onnxruntime_inference_collection.InferenceSession,
        image: np.ndarray,
        window_size: int = 256,
        batch_size: int = 32,
        crop_size: int = 16,
        use_tta: bool = False,
        progress_bar: callable = None) -> np.ndarray:
    """Window inference method

    Args:
        model (ort.capi.onnxruntime_inference_collection.InferenceSession): ORT model
        image (np.ndarray): Image in uint8 format with (H, W) shape
        window_size (int, optional): Window size. Defaults to 256.
        batch_size (int, optional): Batch size to inference. Defaults to 32.
        crop_size (int, optional): Size of window overlapping. Defaults to 0.
        use_tta (bool, optional): Use test time augmentations. Defaults to False.
        progress_bar (callable, optional): Progress bar. Defaults to None.

    Returns:
        np.ndarray: denoised result
    """
    crop_d = crop_size
    output_size = window_size - crop_d * 2

    d = (window_size - output_size) // 2

    margin_width = (
        output_size - image.shape[1] % output_size
    ) * (image.shape[1] % output_size != 0)

    margin_height = (
        output_size - image.shape[0] % output_size
    ) * (image.shape[0] % output_size != 0)

    padded_image = np.pad(
        image,
        ((d, d + margin_height), (d, d + margin_width)),
        mode='reflect'
    )

    # Use CxHxW format
    padded_image = np.expand_dims(padded_image, axis=0)
    # To 0..1
    padded_image = padded_image.astype(np.float32) / 255.0

    predicted_images = []

    transforms = [TensorRotate.NONE]
    if use_tta:
        transforms += [
            TensorRotate.HORISONTAL_FLIP,
            TensorRotate.ROTATE_90_CLOCKWISE,
            TensorRotate.ROTATE_180,
            TensorRotate.ROTATE_90_COUNTERCLOCKWISE
        ]

    transforms_loop_generator = progress_bar(transforms) if progress_bar is not None else transforms

    for transform in transforms_loop_generator:
        transform_padded_image = rotate_tensor(padded_image, transform)

        crops = []
        for i in range(transform_padded_image.shape[1] // output_size):
            for j in range(transform_padded_image.shape[2] // output_size):
                crops.append(
                    transform_padded_image[
                        :,
                        i*output_size:i*output_size + window_size,
                        j*output_size:j*output_size + window_size
                    ]
                )

        crops = array_split(np.stack(crops, axis=0), batch_size)

        outs = np.concatenate(
            [
                run_onnx_model_on_batch(model, bc)[:, :, crop_d:-crop_d, crop_d:-crop_d] \
                    if crop_d > 0 else \
                        run_onnx_model_on_batch(model, bc)
                for bc in crops
            ],
            axis=0
        )

        result_transform_image = outs.reshape(
            transform_padded_image.shape[1] // output_size,
            transform_padded_image.shape[2] // output_size, 
            padded_image.shape[0],
            output_size,
            output_size
        )
        result_transform_image = np.concatenate(
            tuple(np.concatenate(tuple(result_transform_image), axis=2)),
            axis=2
        )

        predicted_images.append(result_transform_image)

    predicted_images = np.stack(
        [
            rotate_tensor(predicted_images[i], transform)
            for i, transform in enumerate(transforms[:2] + transforms[2:][::-1])
        ],
        axis=0
    )
    result_image = np.mean(predicted_images, axis=0)
    result_image = np.clip(result_image, 0.0, 1.0) * 255.0
    result_image = result_image.astype(np.uint8)

    result_img = result_image[
                    :,
                    :image.shape[0],
                    :image.shape[1]
                ]
    return result_img[0]


@st.cache_resource()
def cached_model() -> ort.capi.onnxruntime_inference_collection.InferenceSession:
    providers = ['CPUExecutionProvider']
    # providers = ['CUDAExecutionProvider'] --- if need GPU

    ort_model = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=providers
    )

    return ort_model


def main():
    st.set_page_config(page_title="MRI Image denoising")
    st.set_option("deprecation.showfileUploaderEncoding", False)


    # title area
    st.markdown("""
        # MRI Image Denoising Demo
        > Powered by [Alexey Kovalenko](https://github.com/AlexeySrus)
        """)
    

    model_session = cached_model()


    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp", "dcm"])
    if uploaded_file is not None:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print('Request: {}'.format(dt_string))

        if os.path.splitext(uploaded_file.name)[1].lower() == '.dcm':
            ds = dicom.dcmread(uploaded_file)
            pixel_array_numpy = ds.pixel_array.astype(np.float32)
            pixel_array_numpy = (pixel_array_numpy - pixel_array_numpy.min()) / (pixel_array_numpy.max() - pixel_array_numpy.min() + 1E-5)
            pixel_array_numpy = pixel_array_numpy * 255.0
            pixel_array_numpy = pixel_array_numpy.astype(np.uint8)
            original_image = Image.fromarray(pixel_array_numpy, mode="L")
        else:
            original_image = Image.open(uploaded_file)
            original_image = ImageOps.exif_transpose(original_image)

        original_image = original_image.convert('L')

        denoised_image = window_denoise_inference(model_session, np.array(original_image))

        result = Image.fromarray(denoised_image, mode="L")

        image_comparison(
            img1=original_image,
            img2=result,
            label1="Original Image",
            label2="Denoised Image",
        )

        img_io = io.BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)

        st.download_button(
            'Download denoised PNG image',
            data=img_io.read(),
            file_name='denoised_' + os.path.splitext(uploaded_file.name)[0] + '.png'
        )


if __name__ == "__main__":
    main()
