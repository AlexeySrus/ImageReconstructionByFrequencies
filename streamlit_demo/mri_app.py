import os
import sys
from pathlib import Path

import numpy as np
import cv2
from datetime import datetime
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
MODEL_PATH: str = os.path.join(ROOT_PATH, 'materials/_mri_model.pt')
DEVICE: str = 'cuda:0'
IMAGE_SIZE: int = 256

sys.path.insert(0, RESEARCH_PATH)
from FFTCNN.combined_attn_unet import FFTAttentionUNet as DenoisingModel
from utils.window_inference import eval_denoise_inference


def inference(net: torch.nn.Module, _input: np.ndarray) -> np.ndarray:
    img = np.expand_dims(_input, axis=2)
    input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)

    with torch.no_grad():
        restored_image = eval_denoise_inference(
            tensor_img=input_tensor, model=net, window_size=IMAGE_SIZE, 
            batch_size=4, crop_size=IMAGE_SIZE // 32, use_tta=True, device=DEVICE,
            progress_bar=stqdm
        )

    input_tensor = input_tensor.to('cpu')

    pred_image = restored_image.to('cpu')
    pred_image = torch.clamp(pred_image, 0, 1)
    pred_image = pred_image.permute(1, 2, 0).numpy()
    pred_image = (pred_image * 255.0).astype(np.uint8)

    return pred_image[..., 0]


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
    model = DenoisingModel(in_ch=1, out_ch=1, image_size=IMAGE_SIZE, use_substraction=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE))['model'])
    model.eval()
    
    return model


@st.cache_resource()
def cached_data():
    return UserImagesStorage(root_folder=os.path.join(ROOT_PATH, 'materials/users_images/'), image_ext='.png')


def main():
    st.set_page_config(page_title="MRI Image denoising")
    st.set_option("deprecation.showfileUploaderEncoding", False)


    # title area
    st.markdown("""
        # MRI Image Denoising Demo
        > Powered by [Alexey Kovalenko](https://github.com/AlexeySrus)
        """)

    os.makedirs(
        os.path.join(ROOT_PATH, 'materials/users_images/'),
        exist_ok=True
    )

    images_storage = cached_data()
    model_session = cached_sesstion()
    

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

        images_storage.upload(original_image)

        original_image = original_image.convert('L')

        denoised_image = inference(model_session, np.array(original_image))

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
