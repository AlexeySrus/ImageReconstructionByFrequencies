import logging
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from typing import Tuple
import tqdm
import torch
from torch.utils import data
import torchvision
import segmentation_models_pytorch as smp
import timm
import os
from torchmetrics.image import PeakSignalNoiseRatio as TorchPSNR
from pytorch_msssim import SSIM

from dataloader import SeriesAndComputingClearDataset, PairedDenoiseDataset, SyntheticNoiseDataset
from callbacks import VisImage, VisPlot
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from IS_Net.isnet import ISNetDIS
from utils.window_inference import denoise_inference
from utils.hist_loss import HistLoss


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class CustomTrainingPipeline(object):
    def __init__(self,
                 train_data_paths: Tuple[str, str],
                 val_data_paths: Tuple[str, str],
                 synth_data_paths: str,
                 experiment_folder: str,
                 model_name: str = 'resnet18',
                 load_path: str = None,
                 visdom_port: int = 9000,
                 batch_size: int = 32,
                 epochs: int = 200,
                 resume_epoch: int = 1,
                 stop_criteria: float = 1E-7,
                 device: str = 'cuda',
                 image_size: int = 224,
                 train_workers: int = 4):
        """
        Train model
        Args:
            data_tar_path: Path to training data tar archive
            val_split: Validation part rate
            experiment_folder: Path to folder with checkpoints and experiments data
            model_name: timm classification model name
            load_path: Path to model weights to load
            visdom_port: Port of visualization
            batch_size: Training batch size
            epochs: Count of epoch
            stop_criteria: criteria to stop of training process
            device: Target device to train
            image_size: Input image size
        """
        self.device = device
        self.experiment_folder = experiment_folder
        self.checkpoints_dir = os.path.join(experiment_folder, 'checkpoints/')
        self.output_val_images_dir = os.path.join(experiment_folder, 'val_outputs/')

        self.load_path = load_path
        self.visdom_port = visdom_port  # Set None to disable
        self.batch_size = batch_size
        self.epochs = epochs
        self.resume_epoch = resume_epoch
        self.stop_criteria = stop_criteria
        self.best_test_score = 0

        self.image_shape = (image_size, image_size)

        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.output_val_images_dir, exist_ok=True)

        self.train_base_dataset = SeriesAndComputingClearDataset(
            images_series_folder=train_data_paths[0],
            clear_images_path=train_data_paths[1],
            window_size=self.image_shape[0],
            dataset_size=10000
        )

        self.val_dataset = PairedDenoiseDataset(
            noisy_images_path=val_data_paths[0],
            clear_images_path=val_data_paths[1],
            preload=True
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_base_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_workers,
            pin_memory=True
        )

        self.batch_visualizer = None if visdom_port is None else VisImage(
            title='Denoising',
            port=visdom_port,
            vis_step=150,
            scale=1
        )

        self.plot_visualizer = None if visdom_port is None else VisPlot(
            title='Training curves',
            port=visdom_port
        )

        if self.plot_visualizer is not None:
            self.plot_visualizer.register_scatterplot(
                name='train validation loss per_epoch',
                xlabel='Epoch',
                ylabel='CrossEntropy',
                legend=['train', 'val']
            )

            self.plot_visualizer.register_scatterplot(
                name='validation acc per_epoch',
                xlabel='Epoch',
                ylabel='PSNR',
                legend=['val']
            )

        self.model = ISNetDIS(in_ch=3, out_ch=4 * 3)
        self.dwt = DWT_2D('haar')
        self.iwt = IDWT_2D('haar')
        self.model = self.model.to(device)
        self.optimizer = torch.optim.RAdam(params=self.model.parameters(), lr=0.002)

        if load_path is not None:
            load_data = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(load_data['model'])
            self.optimizer.load_state_dict(load_data['optimizer'])

            print(
                'Model and optimizer have been loaded by path: {}'.format(load_path)
            )

        self.images_criterion = torch.nn.MSELoss(size_average=True)
        self.ssim_loss = SSIMLoss()
        self.hist_loss = HistLoss(image_size=self.image_shape[0], device=self.device)
        self.wavelets_criterion = torch.nn.SmoothL1Loss(size_average=True)
        self.accuracy_measure = TorchPSNR().to(device)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100, 200, 300],
            gamma=0.5
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _compute_wavelets_loss(self, pred_wavelets_pyramid, gt_image, factor: float = 0.9):
        gt_d0_ll = gt_image
        _loss = None
        _h_loss = None
        _loss_scale = 1.0

        for i in range(len(pred_wavelets_pyramid)):
            gt_ll, gt_lh, gt_hl, gt_hh = self.dwt(gt_d0_ll)
            gt_wavelets = torch.cat((gt_ll, gt_lh, gt_hl, gt_hh), dim=1)

            if i == 0:
                _loss = self.wavelets_criterion(pred_wavelets_pyramid[i], gt_wavelets) * _loss_scale * 0.5
                _h_loss = self.hist_loss(pred_wavelets_pyramid[i][:, :3] / 2, gt_ll / 2)
            else:
                _loss += self.wavelets_criterion(pred_wavelets_pyramid[i], gt_wavelets) * _loss_scale
                if i < 3:
                    _h_loss += self.hist_loss(pred_wavelets_pyramid[i][:, :3] / 2, gt_ll / 2) * _loss_scale

            _loss_scale *= factor

            gt_d0_ll = gt_ll / 2.0

        return _loss, _h_loss

    def _train_step(self, epoch) -> float:
        self.model.train()
        avg_epoch_loss = 0

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (_noisy_image, _clear_image) in enumerate(self.train_dataloader):


                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device)

                self.optimizer.zero_grad()
                pred_wavelets_pyramid, _ = self.model(noisy_image)
                pred_ll, pred_lh, pred_hl, pred_hh = torch.split(pred_wavelets_pyramid[0], 3, dim=1)

                restored_image = self.iwt(pred_ll, pred_lh, pred_hl, pred_hh)

                loss = self.images_criterion(restored_image, clear_image)   # / 2 + self.ssim_loss(restored_image, clear_image) / 2
                wloss, hist_loss = self._compute_wavelets_loss(pred_wavelets_pyramid, clear_image)

                total_loss = loss + wloss + hist_loss * 0.01

                total_loss.backward()
                self.optimizer.step()

                pbar.postfix = \
                    'Epoch: {}/{}, px_loss: {:.7f}, w_loss: {:.7f}, h_loss: {:.7f}'.format(
                        epoch,
                        self.epochs,
                        loss.item(),
                        wloss.item(),
                        hist_loss.item()
                    )
                avg_epoch_loss += loss.item() / len(self.train_dataloader)

                if self.batch_visualizer is not None:
                    wavelets_pred = pred_wavelets_pyramid[0].detach()
                    with torch.no_grad():
                        wavelets_gt = self.dwt(clear_image)
                        wavelets_inp = self.dwt(noisy_image)
                        self.batch_visualizer.per_batch(
                            {
                                'input_img': noisy_image,
                                'input_wavelets': wavelets_inp,
                                'pred_image': restored_image.detach(),
                                'pred_wavelets': wavelets_pred.detach(),
                                'gt_wavelets': wavelets_gt,
                                'gt_image': clear_image
                            }
                        )

                pbar.update(1)

        return avg_epoch_loss

    def _validation_step(self) -> Tuple[float, float]:
        self.model.eval()
        avg_acc_rate = 0
        avg_loss_rate = 0
        test_len = 0

        if self.val_dataset is not None:
            for sample_i in tqdm.tqdm(range(len(self.val_dataset))):
                _noisy_image, _clear_image = self.val_dataset[sample_i]

                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device)

                with torch.no_grad():
                    restored_image = denoise_inference(
                        tensor_img=noisy_image, model=self.model, iwt=self.iwt, window_size=self.image_shape[0], batch_size=4
                    ).unsqueeze(0)

                    loss = self.images_criterion(restored_image, clear_image.unsqueeze(0))   #  / 2 + self.ssim_loss(restored_image, clear_image) / 2
                    avg_loss_rate += loss.item()

                    val_psnr = self.accuracy_measure(
                        restored_image,
                        clear_image.unsqueeze(0)
                    )

                    acc_rate = val_psnr.item()

                    avg_acc_rate += acc_rate
                    test_len += 1

                    result_path = os.path.join(self.output_val_images_dir, '{}.png'.format(sample_i + 1))
                    val_img = (torch.clip(restored_image[0].to('cpu').permute(1, 2, 0), 0, 1) * 255.0).numpy().astype(np.uint8)
                    cv2.imwrite(result_path, cv2.cvtColor(val_img, cv2.COLOR_RGB2BGR))

        if test_len > 0:
            avg_acc_rate /= test_len
            avg_loss_rate /= test_len

        self.scheduler.step(avg_loss_rate)

        return avg_loss_rate, avg_acc_rate

    def _plot_values(self, epoch, avg_train_loss, avg_val_loss, avg_val_acc):
        if self.plot_visualizer is not None:
            self.plot_visualizer.per_epoch(
                {
                    'n': epoch,
                    'val loss': avg_val_loss,
                    'loss': avg_train_loss,
                    'val acc': avg_val_acc
                }
            )

    def _save_best_traced_model(self, save_path: str):
        traced_model = torch.jit.trace(self.model, torch.rand(1, 3, *self.image_shape))
        torch.jit.save(traced_model, save_path)

    def _save_best_checkpoint(self, epoch, avg_acc_rate):
        best_model_path = os.path.join(
            self.checkpoints_dir,
            'best.trh'
        )
        latest_model_path = os.path.join(
            self.checkpoints_dir,
            'last.trh'
        )

        self.model.eval()
        save_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(
            save_state,
            latest_model_path
        )

        if self.best_test_score - avg_acc_rate < -1E-5:
            self.best_test_score = avg_acc_rate

            torch.save(
                save_state,
                best_model_path
            )

    def _check_stop_criteria(self):
        return self.get_lr() - self.stop_criteria < -1E-9

    def fit(self):
        for epoch_num in range(self.resume_epoch, self.epochs + 1):
            epoch_train_loss = self._train_step(epoch_num)
            val_loss, val_acc = self._validation_step()
            self._plot_values(epoch_num, epoch_train_loss, val_loss, val_acc)
            self._save_best_checkpoint(epoch_num, val_acc)

            if self._check_stop_criteria():
                break


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Training pipeline')
    parser.add_argument(
        '--experiment_folder', type=str, required=True,
        help='Path to folder with checkpoints and experiments data.'
    )
    parser.add_argument(
        '--epochs', type=int, required=False, default=200
    ),
    parser.add_argument(
        '--image_size', type=int, required=False, default=224
    ),
    parser.add_argument(
        '--resume_epoch', type=int, required=False, default=1
    ),
    parser.add_argument(
        '--load_path', type=str, required=False,
        help='Path to model weights to load.'
    )
    parser.add_argument(
        '--visdom_port', type=int, required=False, default=9000,
        help='Port of visualization.'
    )
    parser.add_argument(
        '--batch_size', type=int, required=False, default=32,
        help='Training batch size.'
    )
    parser.add_argument(
        '--model', type=str, required=False, default='efficientnetv2_s',
        help='Name of model from timm models pool.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_data = (
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_train/images_series/',
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_train/averaged_outclass_series/'
    )

    val_data = (
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/noisy/',
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/clear/'
    )
    clear_path = '/media/alexey/SSDData/datasets/room_inpainting/train/images/'

    CustomTrainingPipeline(
        train_data_paths=train_data,
        val_data_paths=val_data,
        synth_data_paths=clear_path,
        experiment_folder=args.experiment_folder,
        load_path=args.load_path,
        visdom_port=args.visdom_port,
        epochs=args.epochs,
        resume_epoch=args.resume_epoch,
        batch_size=args.batch_size,
        model_name=args.model,
        image_size=args.image_size,
        train_workers=8
    ).fit()

    exit(0)
