from typing import Tuple
from argparse import ArgumentParser, Namespace
import cv2
import numpy as np
import torch
import h5py
from tqdm import tqdm
import torch
import os
import scipy.io as sio
from timeit import default_timer as time

CURRENT_PATH = os.path.dirname(__file__)

from WTSNet.wtsnet import WTSNet, convert_weights_from_old_version
from WTSNet.wts_timm import WTSNetTimm, UnetTimm, SharpnessHead
from utils.window_inference import eval_denoise_inference


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Plot wavelets')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '-n', '--name', type=str, required=False, default='resnet10t',
        help='Timm model name'
    )
    parser.add_argument(
        '-f', '--folder', type=str, required=True,
        help='Path to folder with the Darmstadt Noise Dataset'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=False,
        help='Path to folder with output visualizations (optional)'
    )
    parser.add_argument(
        '--use_unet', action='store_true',
        help='Use classic U-Net network decoder architecture'
    )
    parser.add_argument(
        '--use_tta', action='store_true',
        help='Use test time augmentations until inference'
    )
    parser.add_argument(
        '--use_sharpness_head', action='store_true',
        help='Use sharpness head'
    )
    return parser.parse_args()


def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising
    
    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object_)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )
    

def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy()
    _img = (_img * 255.0).astype(np.uint8)
    return _img


def tensor_to_srgb_matrix(t: torch.Tensor) -> np.ndarray:
    _img = t.permute(1, 2, 0).numpy().astype(np.float32)
    return _img


if __name__ == '__main__':
    args = parse_args()

    imgsz = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Device: {}'.format(device))

    load_path = args.model
    load_data = torch.load(load_path, map_location=device)

    if args.use_unet:
        try:
            model = UnetTimm(model_name=args.name, use_biliniar=False).to(device)
            model.load_state_dict(load_data['model'])
        except:
            model = UnetTimm(model_name=args.name, use_biliniar=True).to(device)
            model.load_state_dict(load_data['model'])
            print('Use U-Net with bilinear')
    else:
        model = WTSNetTimm(model_name=args.name, use_clipping=True).to(device)

    if args.use_sharpness_head:
        try:
            model = SharpnessHead(
                base_model=model,
                in_ch=3, out_ch=3
            ).to(device)
            model.load_model(load_data)
            print('Used full-channels sharpness head')
        except:
            model = SharpnessHeadForYChannel(
                base_model=model,
                in_ch=3, out_ch=3
            ).to(device)
            model.load_model(load_data)
            print('Used Y-channel sharpness head')
    else:
        model.load_state_dict(load_data['model'])

    model.eval()

    # Fake run to optimize
    with torch.no_grad():
        _ = model(torch.rand(1, 3, imgsz, imgsz).to(device))

    print('Best torchmetric PSNR: {:.2f}'.format(load_data['acc']))

    if args.output is None:
        output_folder = os.path.join(CURRENT_PATH, '../../../materials/dnd_inference_results/')
    else:
        output_folder = args.output

    os.makedirs(output_folder, exist_ok=True)

    dnd_folder = args.folder
    matrices_folder = os.path.join(dnd_folder, 'images_srgb/')

    info = h5py.File(os.path.join(dnd_folder, 'info.mat'), 'r')['info']
    bb = info['boundingboxes']

    inference_time_series = []

    for image_idx in tqdm(range(50)):
        image_name = '%04d.mat' % (image_idx + 1)
        image_path = os.path.join(matrices_folder, image_name)

        img = h5py.File(image_path, 'r')
        img = np.float32(np.array(img['InoisySRGB']).T)

        img = cv2.cvtColor(
            (img * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2YCrCb
        ).astype(np.float32) / 255.0

        boxes = np.array(info[bb[0][image_idx]]).T
        for k in range(len(boxes)):
            box = [
                int(boxes[k, 0] - 1),
                int(boxes[k, 2]),
                int(boxes[k, 1] - 1),
                int(boxes[k, 3])
            ]
            crop_img = img[box[0]:box[1], box[2]:box[3]].copy()

            input_tensor = torch.from_numpy(crop_img.astype(np.float32).transpose((2, 0, 1)))
            
            start_inference_time = time()
            with torch.no_grad():
                restored_image = eval_denoise_inference(
                    tensor_img=input_tensor, model=model, window_size=imgsz, 
                    batch_size=4, crop_size=imgsz // 32, use_tta=args.use_tta, device=device
                )
            finish_inference_time = time()
            crop_infernece_time = finish_inference_time - start_inference_time
            inference_time_series.append(crop_infernece_time)

            input_tensor = input_tensor.to('cpu')

            pred = restored_image.to('cpu')
            pred = torch.clamp(pred, 0, 1)
            pred_image = tensor_to_image(pred)

            pred_image = cv2.cvtColor(pred_image, cv2.COLOR_YCrCb2RGB)
            pred_matrix = pred_image.astype(np.float32) / 255.0

            output_matrix_name = '%04d_%02d.mat' % (image_idx + 1, k + 1)
            out_matrix_path = os.path.join(
                output_folder,
                output_matrix_name
            )
            sio.savemat(out_matrix_path, {'Idenoised_crop': pred_matrix})

            out_image_path = os.path.join(
                output_folder,
                os.path.splitext(output_matrix_name)[0] + '.png'
            )
            cv2.imwrite(
                out_image_path,
                cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
            )

        print('Mean crop inference time: {}'.format(np.array(inference_time_series).mean()))

    bundle_submissions_srgb(output_folder)
    
