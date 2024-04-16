import os
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageOps
import pydicom as dicom
import base64
import torch
import io

import streamlit as st
from stqdm import stqdm
from streamlit_image_comparison import image_comparison

ROOT_PATH: str = os.path.join(os.path.dirname(__file__), '../')
RESEARCH_PATH: str = os.path.join(ROOT_PATH, 'research_pipelines/denoising_with_fourier/')
MODEL_PATH: str = os.path.join(ROOT_PATH, 'materials/unet.pt')
DEVICE: str = 'cuda:0'
IMAGE_SIZE: int = 256
USE_TTA: bool = False
USE_YCRCB_COLOR_SPACE: bool = False
USE_UNET_PLUS_PLUS: bool = False

if USE_UNET_PLUS_PLUS:
    MODEL_PATH: str = os.path.join(ROOT_PATH, 'materials/unet_plus_plus.pt')

sys.path.insert(0, RESEARCH_PATH)
from FFTCNN.combined_attn_unet import FFTAttentionUNet as DenoisingModel
from FFTCNN.combined_attn_unet_plusplus import FFTAttentionUNetPlusPlus as DenoisingModelPlusPlus
from utils.window_inference import eval_denoise_inference
from utils.tensor_utils import convert_tensor_to_rgb, convert_tensor_to_ycrcb_or_grayscale


def inference(net: torch.nn.Module, _input: np.ndarray) -> np.ndarray:
    img = _input.copy()
    input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)
    input_tensor = convert_tensor_to_ycrcb_or_grayscale(input_tensor.unsqueeze(0), USE_YCRCB_COLOR_SPACE, False)[0]

    with torch.no_grad():
        restored_image = eval_denoise_inference(
            tensor_img=input_tensor, model=net, window_size=IMAGE_SIZE, 
            batch_size=4, crop_size=IMAGE_SIZE // 32, use_tta=USE_TTA, device=DEVICE,
            progress_bar=stqdm
        )


    input_tensor = input_tensor.to('cpu')
    del input_tensor

    pred_image = restored_image.to('cpu')
    pred_image = convert_tensor_to_rgb(pred_image.unsqueeze(0), USE_YCRCB_COLOR_SPACE, False)[0]
    pred_image = torch.clamp(pred_image, 0, 1)
    pred_image = pred_image.permute(1, 2, 0).numpy()
    pred_image = (pred_image * 255.0).astype(np.uint8)

    return pred_image


class UserImagesStorage(object):
    def __init__(self, root_folder: str, image_ext: str = '.webp'):
        assert os.path.isdir(root_folder), '{} is not exists'.format(root_folder)
        assert image_ext.startswith('.'), \
            'Invalid extension \'{}\', correct example: \'.webp\''.format(image_ext)

        self.root_folder = root_folder
        self.image_ext = image_ext

        exist_files = os.listdir(self.root_folder)

        self.last_number = 1
        if len(exist_files) > 0:
            self.last_number = max(
                [int(os.path.splitext(fname)[0]) for fname in exist_files]
            ) + 1

    def upload(self, image: Image):
        save_path = os.path.join(
            self.root_folder,
            '{}{}'.format(self.last_number, self.image_ext)
        )
        image.save(save_path)

        self.last_number += 1


@st.cache_resource()
def cached_sesstion():
    if USE_UNET_PLUS_PLUS:
        print('Use U-Net++')
        model = DenoisingModelPlusPlus(image_size=IMAGE_SIZE).to(DEVICE)
    else:
        print('Use U-Net')
        model = DenoisingModel(image_size=IMAGE_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE))['model'])
    model.eval()
    
    return model


@st.cache_resource()
def cached_data():
    return UserImagesStorage(root_folder=os.path.join(ROOT_PATH, 'materials/users_images/'), image_ext='.png')


def main():
    st.set_page_config(page_title="Image denoising")
    st.set_option("deprecation.showfileUploaderEncoding", False)


    # title area
    st.markdown("""
        # Image Denoising Demo
        > Powered by [Alexey Kovalenko](https://github.com/AlexeySrus)
        """)

    os.makedirs(
        os.path.join(ROOT_PATH, 'materials/users_images/'),
        exist_ok=True
    )

    images_storage = cached_data()
    model_session = cached_sesstion()
    

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        original_image = ImageOps.exif_transpose(original_image)

        images_storage.upload(original_image)

        denoised_image = inference(model_session, np.array(original_image))

        result =Image.fromarray(denoised_image, mode="RGB")

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
