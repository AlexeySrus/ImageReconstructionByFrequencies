from typing import Tuple, Optional
import logging
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import tqdm
import torch
from torch.utils import data
import torchvision
import segmentation_models_pytorch as smp
import timm
from PIL import Image
import os
from torchmetrics.image import PeakSignalNoiseRatio as TorchPSNR
from pytorch_msssim import SSIM, MS_SSIM
from piq import DISTS
from pytorch_optimizer import AdaSmooth, Ranger21
from utils.haar_utils import HaarForward, HaarInverse

from dataloader import PairedDenoiseDataset, SyntheticNoiseDataset
from callbacks import VisImage, VisAttentionMaps, VisPlot, SaveTableTrainInfo
from WTSNet.wts_timm import UnetTimm, WTSNetTimm
from utils.window_inference import denoise_inference
from utils.hist_loss import HistLoss
from utils.freq_loss import HFENLoss
from utils.sparce_loss import SparceLoss
from utils.adversarial_loss import Adversarial
from utils.uformer_model import get_uformer_model
from utils.tv_loss import TVLoss, CharbonnierLoss
from utils.tensor_utils import MixUp_AUG


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class MIXLoss(MS_SSIM):
    base_loss = CharbonnierLoss()
    def forward(self, x, y):
        return (1. - super().forward(x, y)) * (1-0.84) + self.base_loss(x, y) * 0.84
    

class DWTHaar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = HaarForward()

    def forward(self, x):
        out = self.dwt(x)
        step = out.size(1) // 4
        ll = out[:, :step]
        lh = out[:, step:step*2]
        hl = out[:, step*2:step*3]
        hh = out[:, step*3:]
        return [ll, lh, hl, hh]
    

class IWTHaar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.iwt = HaarInverse()

    def forward(self, ll, lh, hl, hh):
        return self.iwt(torch.cat((ll, lh, hl, hh), dim=1))


class CustomTrainingPipeline(object):
    def __init__(self,
                 train_data_paths: Tuple[str, str],
                 val_data_paths: Tuple[str, str],
                 synth_data_paths: Optional[str],
                 experiment_folder: str,
                 load_path: Optional[str] = None,
                 model_name: str = 'resnet10t',
                 visdom_port: int = 9000,
                 batch_size: int = 32,
                 epochs: int = 200,
                 stop_criteria: float = 1E-7,
                 device: str = 'cuda',
                 image_size: int = 512,
                 train_workers: int = 0,
                 preload_data: bool = False,
                 lr_steps: int = 4,
                 no_load_optim: bool = False,
                 gradient_accumulation_steps: int = 1,
                 annottaion_str: str = ''):
        """
        Train U-Net denoising model

        Args:
            train_data_paths (Tuple[str, str]): Pair of paths to noisy images and clear images
            val_data_paths (Tuple[str, str]): Pair of paths to noisy images and clear images
            synth_data_paths (str, optional): Deprecated parameter, need set as None
            experiment_folder (str): Path to folder with checkpoints and experiments data
            load_path (str, optional): Path to model weights to load. Defaults to None.
            visdom_port (int, optional): Port of visualization. Defaults to 9000.
            batch_size (int, optional): Training batch size. Defaults to 32.
            epochs (int, optional): Count of epoch. Defaults to 200.
            stop_criteria (float, optional): Criteria to stop of training process. Defaults to 1E-7.
            device (str, optional): Target device to train. Defaults to 'cuda'.
            image_size (int, optional): Input image size. Defaults to 512.
            train_workers (int, optional): Count of parallel dataloaders. Defaults to 0.
            preload_data (bool, optional): Load training and validation data to RAM. Defaults to False.
            lr_steps (int, optional): Count of uniformed LR steps. Defaults to 4.
            no_load_optim (bool, optional): Disable load optimizer from checkpoint. Defaults to False.
            gradient_accumulation_steps (int, optional): Count of saved gradients. Defaults to 1.
            annottaion_str (str, optional): Annotation string of experiment. Defaults to ''.
        """
        self.device = device
        self.experiment_folder = experiment_folder
        self.checkpoints_dir = os.path.join(experiment_folder, 'checkpoints/')
        self.output_val_images_dir = os.path.join(experiment_folder, 'val_outputs/')
        self.metrics_logging_file = os.path.join(experiment_folder, 'train_logs.csv')
        self.annotation_file = os.path.join(experiment_folder, 'annotation.txt')

        self.load_path = load_path
        self.visdom_port = visdom_port  # Set None to disable
        self.batch_size = batch_size
        self.epochs = epochs
        self.resume_epoch = 1
        self.stop_criteria = stop_criteria
        self.best_test_score = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.image_shape = (image_size, image_size)

        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.output_val_images_dir, exist_ok=True)

        if len(annottaion_str) > 0:
            with open(self.annotation_file, 'w') as f:
                f.write(annottaion_str + '\n')

        self.train_base_dataset = PairedDenoiseDataset(
                noisy_images_path=train_data_paths[0],
                clear_images_path=train_data_paths[1],
                need_crop=True,
                window_size=self.image_shape[0],
                optional_dataset_size=200000,
                preload=preload_data
            )

        if synth_data_paths is not None:
            self.train_synth_dataset = SyntheticNoiseDataset(
                clear_images_path=synth_data_paths,
                window_size=self.image_shape[0],
                preload=preload_data,
                optional_dataset_size=10000
            )

            self.train_base_dataset = torch.utils.data.ConcatDataset(
                [self.train_base_dataset, self.train_synth_dataset]
            )

        self.val_dataset = PairedDenoiseDataset(
            noisy_images_path=val_data_paths[0],
            clear_images_path=val_data_paths[1],
            preload=preload_data,
            return_names=True
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_base_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_workers
        )

        self.train_metrics_logger = SaveTableTrainInfo(
            table_path=self.metrics_logging_file,
            load_existing_table=load_path is not None
        )

        self.images_visualizer = None if visdom_port is None else VisImage(
            title='Denoising',
            port=visdom_port,
            vis_step=150,
            scale=2.0
        )

        self.attention_visualizer = None if visdom_port is None else VisAttentionMaps(
            title='Denoising',
            port=visdom_port,
            vis_step=150,
            scale=1.0,
            maps_count=5
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

            self.plot_visualizer.register_scatterplot(
                name='validation roc per_epoch',
                xlabel='Epoch',
                ylabel='SSIM',
                legend=['val']
            )

        self.model = WTSNetTimm(model_name=model_name)
        # self.model = get_uformer_model(image_size)
        # self.model.apply(init_weights)
        self.dwt = DWTHaar()
        self.iwt = IWTHaar()
        self.model = self.model.to(device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=0.001, weight_decay=0.01)
        # self.optimizer = AdaSmooth(params=self.model.parameters(), lr=0.001)

        if load_path is not None:
            load_data = torch.load(load_path, map_location=self.device)

            self.resume_epoch = load_data['epoch']
            self.model.load_state_dict(load_data['model'])

            print(
                '#' * 5 + ' Model has been loaded by path: {} '.format(load_path) +  '#' * 5
            )

            if not no_load_optim:
                self.optimizer.load_state_dict(load_data['optimizer'])
                print(
                    '#' * 5 + ' Optimizer has been loaded by path: {} '.format(load_path) + '#' * 5
                )
                self.optimizer.param_groups[0]['lr'] = 0.0001
                print('Optimizer LR: {:.5f}'.format(self.get_lr()))

        # self.optimizer = AdaSmooth(params=self.model.parameters(), lr=0.001)
        
        # self.images_criterion = CharbonnierLoss()
        self.images_criterion = MIXLoss(data_range=1.0)
        # self.images_criterion = torch.nn.SmoothL1Loss()
        # self.perceptual_loss = DISTS()
        self.perceptual_loss = None
        # self.final_hist_loss = HistLoss(image_size=128, device=self.device)
        self.final_hist_loss = None
        # self.hight_freq_loss = HFENLoss(loss_f=torch.nn.functional.l1_loss, norm=True)
        self.mixp_aug = MixUp_AUG()

        # self.adverserial_losses = [
        #     Adversarial(image_size=256, in_ch=3 * 3).to(device),
        #     Adversarial(image_size=128, in_ch=3 * 3).to(device),
        #     Adversarial(image_size=64, in_ch=3 * 3).to(device),
        #     Adversarial(image_size=32, in_ch=3 * 4).to(device)
        # ]

        # self.adv_loss = Adversarial(image_size=image_size, in_ch=3).to(device)
        self.wavelets_criterion = torch.nn.functional.l1_loss
        # self.wavelets_criterion = SparceLoss()
        # self.wavelets_criterion = CharbonnierLoss()
        self.accuracy_measure = TorchPSNR().to(device)
        self.ssim_measure = SSIM(data_range=1.0, channel=3)

        self.scheduler = None

        if lr_steps > 0:
            _lr_steps = lr_steps + 1
            lr_milestones = [
                int(i * (epochs / _lr_steps))
                for i in range(1, _lr_steps)
            ]
            print('Leaning rate milestone epochs: {}'.format(lr_milestones))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=lr_milestones,
                gamma=0.1,
                verbose=True
            )
            self.step_scheduler = False
        else:
            self.step_scheduler = True

            # self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            #     self.optimizer, 
            #     base_lr=0.00001, 
            #     max_lr=0.01
            # )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def _add_loss(self, loss1, loss2):
        if loss1 is None:
            return loss2
        return loss1 + loss2

    def _compute_wavelets_loss(self, pred_wavelets_pyramid, gt_image, factor: float = 1.0, use_approximation: bool = True):
        gt_d0_ll = gt_image
        _loss = None
        _loss_scale = 1.0

        for i in range(len(pred_wavelets_pyramid)):
            gt_ll, gt_lh, gt_hl, gt_hh = self.dwt(gt_d0_ll)
            
            gt_wavelets = torch.cat((gt_ll, gt_lh, gt_hl, gt_hh), dim=1)
            if use_approximation:
                _loss = self._add_loss(_loss, self.wavelets_criterion(pred_wavelets_pyramid[i], gt_wavelets)) * _loss_scale
            else:
                _loss = self._add_loss(_loss, self.wavelets_criterion(pred_wavelets_pyramid[i][:, 3:], gt_wavelets[:, 3:])) * _loss_scale

            _loss_scale *= factor

            gt_d0_ll = gt_ll

        return _loss / len(pred_wavelets_pyramid)

    def _compute_deep_iwt_loss(self, pred_wavelets_pyramid, pred_image, input_image):
        composed_image = pred_wavelets_pyramid[-1][:, :3]

        _loss = self.images_criterion(
            composed_image,
            torch.nn.functional.interpolate(input_image, (composed_image.size(2), composed_image.size(3)), mode='area')
        )

        for i in range(len(pred_wavelets_pyramid) - 1, -1, -1):
            la, lh, hl, hh = torch.split(pred_wavelets_pyramid[i], 3, dim=1)
            if i < len(pred_wavelets_pyramid) - 1:
                _loss += self.images_criterion(composed_image, la)
            composed_image = self.iwt(composed_image, lh, hl, hh)

        _loss += self.images_criterion(composed_image, pred_image)

        return _loss / (len(pred_wavelets_pyramid) + 1)

    def _compute_adversarial_loss(self, pred_wavelets_pyramid, gt_image, factor: float = 1.0):
        gt_d0_ll = gt_image
        _loss = 0.0
        _loss_scale = 1.0

        for i in range(len(pred_wavelets_pyramid)):
            gt_ll, gt_lh, gt_hl, gt_hh = self.dwt(gt_d0_ll)
            target_loss_function = self.adverserial_losses[i]

            gt_wavelets = torch.cat((gt_lh, gt_hl, gt_hh), dim=1)
            _loss += target_loss_function(pred_wavelets_pyramid[i][:, 3:], gt_wavelets) * _loss_scale

            _loss_scale *= factor

            gt_d0_ll = gt_ll

        return _loss

    def _train_step(self, epoch) -> float:
        self.model.train()
        avg_epoch_loss = 0

        batches_count = len(self.train_dataloader)

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for idx, (_noisy_image, _clear_image) in enumerate(self.train_dataloader):
                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device)

                if epoch >= 5:
                    if epoch > 1 and np.random.randint(0, 101) > 80:
                        clear_image, noisy_image = self.mixp_aug.aug(clear_image, noisy_image)

                # self.optimizer.zero_grad()
                output = self.model(noisy_image)
                pred_image = output[0]
                pred_wavelets_pyramid = output[1]
                spatial_attention_maps = output[2]

                loss = self.images_criterion(pred_image, clear_image)

                # a_loss = self.adv_loss(pred_image, noisy_image)

                if self.perceptual_loss is not None:
                    loss = loss / 2 + self.perceptual_loss(pred_image, clear_image) / 2
                    
                wloss = self._compute_wavelets_loss(pred_wavelets_pyramid, clear_image, use_approximation=False)
                # wloss = self._compute_deep_iwt_loss(pred_wavelets_pyramid, pred_image, noisy_image)
                
                # hist_loss = self.final_hist_loss(pred_image, clear_image)
                hist_loss = 0
                # hf_loss = self.hight_freq_loss(pred_image, clear_image)

                total_loss = wloss

                if self.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.gradient_accumulation_steps

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # self.optimizer.step()

                if (self.gradient_accumulation_steps <= 1) or (
                        (idx + 1) % self.gradient_accumulation_steps == 0) or (
                        idx + 1 == batches_count):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.postfix = \
                    'Epoch: {}/{}, px_loss: {:.7f}, w_loss: {:.7f}'.format(
                        epoch,
                        self.epochs,
                        loss.item(),
                        wloss.item()
                    )
                avg_epoch_loss += loss.item() / len(self.train_dataloader)

                if self.images_visualizer is not None:
                    wavelets_pred = pred_wavelets_pyramid[0].detach()
                    with torch.no_grad():
                        wavelets_gt = self.dwt(clear_image)
                        wavelets_inp = self.dwt(noisy_image)

                        vis_idx = self.images_visualizer.per_batch(
                            {
                                'input_img': noisy_image,
                                'input_wavelets': wavelets_inp,
                                'pred_image': pred_image.detach(),
                                'pred_wavelets': wavelets_pred.detach(),
                                'gt_wavelets': wavelets_gt,
                                'gt_image': clear_image
                            }
                        )

                        if len(spatial_attention_maps) > 0:
                            self.attention_visualizer.per_batch(
                                {
                                    'sa_list': spatial_attention_maps
                                },
                                i=vis_idx
                            )

                if self.scheduler is not None and self.step_scheduler:
                    self.scheduler.step()

                pbar.update(1)

        return avg_epoch_loss

    def _validation_step(self) -> Tuple[float, float]:
        self.model.eval()
        avg_acc_rate = 0
        avg_ssim_rate = 0
        avg_loss_rate = 0
        test_len = 0

        if self.val_dataset is not None:
            for sample_i in tqdm.tqdm(range(len(self.val_dataset))):
                _noisy_image, _clear_image, image_name = self.val_dataset[sample_i]

                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device).unsqueeze(0)

                with torch.no_grad():
                    restored_image = denoise_inference(
                        tensor_img=noisy_image, model=self.model, window_size=self.image_shape[0], 
                        batch_size=self.batch_size, crop_size=self.image_shape[0] // 32
                    ).unsqueeze(0)

                    loss = self.images_criterion(restored_image, clear_image)
                    avg_loss_rate += loss.item()

                    restored_image = torch.clamp(restored_image, 0, 1)

                    val_psnr = self.accuracy_measure(
                        restored_image,
                        clear_image
                    )

                    val_ssim = self.ssim_measure(
                        restored_image,
                        clear_image
                    )

                    acc_rate = val_psnr.item()

                    avg_acc_rate += acc_rate
                    avg_ssim_rate += val_ssim.item()
                    del val_ssim
                    test_len += 1

                    # result_path = os.path.join(self.output_val_images_dir, '{}.png'.format(sample_i + 1))
                    result_path = os.path.join(self.output_val_images_dir, image_name)
                    val_img = (torch.clip(restored_image[0].to('cpu').permute(1, 2, 0), 0, 1) * 255.0).numpy().astype(np.uint8)
                    val_img = cv2.cvtColor(val_img, cv2.COLOR_YCrCb2RGB)
                    Image.fromarray(val_img).save(result_path)

        if test_len > 0:
            avg_acc_rate /= test_len
            avg_loss_rate /= test_len
            avg_ssim_rate /= test_len

        if self.scheduler is not None and not self.step_scheduler:
            self.scheduler.step()

        return avg_loss_rate, (avg_acc_rate, avg_ssim_rate)

    def _plot_values(self, epoch, avg_train_loss, avg_val_loss, avg_val_acc):
        avg_val_psnr, avg_val_ssim = avg_val_acc

        if self.plot_visualizer is not None:
            self.plot_visualizer.per_epoch(
                {
                    'n': epoch,
                    'val loss': avg_val_loss,
                    'loss': avg_train_loss,
                    'val acc': avg_val_psnr,
                    'val roc': avg_val_ssim,
                }
            )

        if self.train_metrics_logger is not None:
            self.train_metrics_logger.per_epoch(
                {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_psnr': avg_val_psnr,
                    'val_ssim': avg_val_ssim
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
            'optimizer': self.optimizer.state_dict(),
            'acc': avg_acc_rate,
            'epoch': epoch
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
            val_loss, val_accs = self._validation_step()
            self._plot_values(epoch_num, epoch_train_loss, val_loss, val_accs)
            self._save_best_checkpoint(epoch_num, val_accs[0])

            if self.scheduler is not None and self._check_stop_criteria():
                break


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Training pipeline')
    parser.add_argument(
        '--model', type=str, required=False, default='resnet10t',
        help='Model name from timm library.'
    )
    parser.add_argument(
        '--train_data_folder', type=str, required=True,
        help='Path folder with train data (contains clear/ and noisy/ subfolders).'
    )
    parser.add_argument(
        '--validation_data_folder', type=str, required=True,
        help='Path folder with validation data (contains clear/ and noisy/ subfolders).'
    )
    parser.add_argument(
        '--experiment_folder', type=str, required=True,
        help='Path to folder with checkpoints and experiments data.'
    )
    parser.add_argument(
        '--epochs', type=int, required=False, default=200
    )
    parser.add_argument(
        '--image_size', type=int, required=False, default=512
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
        '--njobs', type=int, required=False, default=8,
        help='Count of dataset workers.'
    )
    parser.add_argument(
        '--batch_size', type=int, required=False, default=32,
        help='Training batch size.'
    )
    parser.add_argument(
        '--lr_milestones', type=int, required=False, default=3,
        help='Count or learning rate scheduler milestones.'
    )
    parser.add_argument(
        '--grad_accum_steps', type=int, required=False, default=1,
        help='Count of batches to accumulate gradiets.'
    )
    parser.add_argument(
        '--preload_datasets', action='store_true',
        help='Load images from datasaets into memory.'
    )
    parser.add_argument(
        '--no_load_optim', action='store_true',
        help='Disable optimizer parameters loading from checkpoint.'
    )
    parser.add_argument(
        '--synthetic_data_paths', type=str, required=False,
        help='Path to folder with clear images to generate synthetic noisy dataset.'
    )
    parser.add_argument(
        '--annotation', type=str, required=False, default='',
        help='Annotation of experiment.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_data = (
        os.path.join(args.train_data_folder, 'noisy/'),
        os.path.join(args.train_data_folder, 'clear/')
    )
    val_data = (
        os.path.join(args.validation_data_folder, 'noisy/'),
        os.path.join(args.validation_data_folder, 'clear/')
    )

    CustomTrainingPipeline(
        model_name=args.model,
        train_data_paths=train_data,
        val_data_paths=val_data,
        synth_data_paths=args.synthetic_data_paths,
        experiment_folder=args.experiment_folder,
        load_path=args.load_path,
        visdom_port=args.visdom_port,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_workers=args.njobs,
        preload_data=args.preload_datasets,
        lr_steps=args.lr_milestones,
        no_load_optim=args.no_load_optim,
        gradient_accumulation_steps=args.grad_accum_steps,
        annottaion_str=args.annotation
    ).fit()

    exit(0)
