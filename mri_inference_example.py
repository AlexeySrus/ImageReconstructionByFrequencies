import os
import sys

import numpy as np
import cv2
import torch
from tqdm import tqdm


RESEARCH_PATH: str = os.getenv('RESEARCH_PATH')
MODEL_PATH: str = os.getenv('MODEL_PATH')
IMAGE_SIZE: int = 256
DEVICE: str = 'cpu'

sys.path.insert(0, RESEARCH_PATH)
from FFTCNN.uformer import Uformer
from utils.window_inference import eval_denoise_inference


def inference(net: torch.nn.Module, _input: np.ndarray) -> np.ndarray:
    img = np.expand_dims(_input, axis=2)
    input_tensor = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1)) / 255.0)

    with torch.no_grad():
        restored_image = eval_denoise_inference(
            tensor_img=input_tensor, model=net, window_size=IMAGE_SIZE, 
            batch_size=4, crop_size=IMAGE_SIZE // 32, use_tta=True, device=DEVICE,
            progress_bar=tqdm
        )

    input_tensor = input_tensor.to('cpu')

    pred_image = restored_image.to('cpu')
    pred_image = torch.clamp(pred_image, 0, 1)
    pred_image = pred_image.permute(1, 2, 0).numpy()
    pred_image = (pred_image * 255.0).astype(np.uint8)

    return pred_image[..., 0]


def get_model():
    model = Uformer(
        img_size=IMAGE_SIZE, embed_dim=32, win_size=8, 
        token_projection='linear', token_mlp='leff',
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True,
        dd_in=1, in_chans=1,
        attention_mode='ca'
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE))['model'])
    model.eval()
    
    return model


if __name__ == '__main__':
    image_path = 'materials/test_mri_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    model_session = get_model()

    denoised_image = inference(model_session, image)

    cv2.imwrite('materials/result_mri_image.png', denoised_image)
