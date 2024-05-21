import logging
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
import kornia
from typing import Tuple, Optional, Union, List, Callable
import tqdm
import torch
from torch.utils import data
import torchvision
import segmentation_models_pytorch as smp
import timm
import itertools
from PIL import Image
import os
from torchmetrics.image import PeakSignalNoiseRatio as TorchPSNR
from pytorch_msssim import SSIM, MS_SSIM
from piq import DISTS
import yaml
from haar_pytorch import HaarForward, HaarInverse
from FDL_pytorch import FDL_loss
from pytorch_optimizer import AdaSmooth

from dataloader import PairedDenoiseDataset, SyntheticNoiseDataset, SYNTH_CONFIG
from callbacks import VisImage, VisAttentionMaps, VisPlot
from FFTCNN.combined_attn_unet import init_weights
from FFTCNN.combined_attn_unet import FFTAttentionUNet
from FFTCNN.combined_attn_unet_plusplus import FFTAttentionUNetPlusPlus
from FFTCNN.uformer import Uformer
from utils.window_inference import denoise_inference
from utils.hist_loss import HistLoss
from utils.adversarial_loss import Adversarial
from utils.freq_loss import HightFrequencyFFTLoss, HFENLoss
from utils.focal_frequency_loss import FocalFrequencyLoss
from utils.edge_loss import EdgeLoss
from utils.laplassian_loss import LapLoss
from utils.tv_loss import CharbonnierLoss, TVLoss
from utils.tensor_utils import MixUp_AUG, convert_tensor_to_rgb
from utils.cas import contrast_adaptive_sharpening


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
    

def calculate_loss(pred_values, truth_value, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], seq_pred: bool) -> torch.Tensor:
    if seq_pred:
        res_loss = sum(
            [
                loss_function(pred_value, truth_value) / (int(lvl_i > 0) * 10 + lvl_i * 2 + 1)
                for lvl_i, pred_value in enumerate(pred_values)
            ]
        )
    else:
        res_loss = loss_function(pred_values, truth_value)

    return res_loss


class ModelMultitask(torch.nn.Module):
    def __init__(self, losses_count: int):
        super().__init__()
        self.sigma = torch.nn.Parameter(torch.ones(losses_count))
        self.eps = 1e-7
	
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:	
        loss_combine = 0.5 * torch.stack(losses, dim=0) / (self.sigma ** 2 + self.eps)
        loss_combine = loss_combine.sum() + torch.log(self.sigma.prod() + self.eps)
        return loss_combine


class CustomTrainingPipeline(object):
    def __init__(self,
                 train_data_paths: Optional[Tuple[str, str]],
                 val_data_paths: Tuple[str, str],
                 synth_data_paths: Optional[str],
                 experiment_folder: str,
                 load_path: Optional[str] = None,
                 visdom_port: int = 9000,
                 batch_size: int = 32,
                 epochs: int = 200,
                 resume_epoch: int = 1,
                 stop_criteria: float = 1E-7,
                 device: str = 'cuda',
                 image_size: int = 512,
                 train_workers: int = 0,
                 preload_data: bool = False,
                 init_lr: float = 0.001,
                 lr_steps: int = 4,
                 no_load_optim: bool = False,
                 gradient_accumulation_steps: int = 1,
                 annottaion_str: str = '',
                 use_ycrcb: bool = False,
                 attention_mode: str = 'full',
                 grayscale: bool = False,
                 use_unetpp: bool = False,
                 use_uformer: bool = False,
                 substracted_noise: bool = False,
                 full_args: Optional[Namespace] = None):
        """
        Train U-Net denoising model

        Args:
            train_data_paths (Tuple[str, str], optional): Pair of paths to noisy images and clear images
            val_data_paths (Tuple[str, str]): Pair of paths to noisy images and clear images
            synth_data_paths (str, optional): Deprecated parameter, need set as None
            experiment_folder (str): Path to folder with checkpoints and experiments data
            load_path (str, optional): Path to model weights to load. Defaults to None.
            visdom_port (int, optional): Port of visualization. Defaults to 9000.
            batch_size (int, optional): Training batch size. Defaults to 32.
            epochs (int, optional): Count of epoch. Defaults to 200.
            resume_epoch (int, optional): Epoch number to resume training. Defaults to 1.
            stop_criteria (float, optional): Criteria to stop of training process. Defaults to 1E-7.
            device (str, optional): Target device to train. Defaults to 'cuda'.
            image_size (int, optional): Input image size. Defaults to 512.
            train_workers (int, optional): Count of parallel dataloaders. Defaults to 0.
            preload_data (bool, optional): Load training and validation data to RAM. Defaults to False.
            init_lr (float, optional): Start learning rate. Defaults to 0.001.
            lr_steps (int, optional): Count of uniformed LR steps. Defaults to 4.
            no_load_optim (bool, optional): Disable load optimizer from checkpoint. Defaults to False.
            gradient_accumulation_steps (bool, optional): Count of accumulated gradients per train batches.
            annottaion_str (str, optional): Annotation string of experiment. Defaults to ''.
            use_ycrcb (bool, optional): Use YCrCb color space. Defaults to False.
            attention_mode (str, optional): Attention mode. Defaults to 'full'.
            grayscale (bool, optional): Use 1-channel images in pipeline. Defaults to False.
            use_unetpp (bool, optional): Use U-Net++ architecture. Defaults to False.
            use_uformer (bool, optional): Use U-Former architecture. Defaults to False.
            substracted_noise (bool, optinal): Use netwotk prediction as Y = X + F(X). Defaults to False.
            full_args (Namespace, optional): All command-line arguments. Defaules to None.
        """
        self.device = device
        self.experiment_folder = experiment_folder
        self.checkpoints_dir = os.path.join(experiment_folder, 'checkpoints/')
        self.output_val_images_dir = os.path.join(experiment_folder, 'val_outputs/')
        self.annotation_file = os.path.join(experiment_folder, 'annotation.txt')

        self.load_path = load_path
        self.visdom_port = visdom_port  # Set None to disable
        self.batch_size = batch_size
        self.epochs = epochs
        self.resume_epoch = resume_epoch
        self.stop_criteria = stop_criteria
        self.best_test_score = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_ycrcb = use_ycrcb
        self.grayscale = grayscale
        self.use_unetpp = use_unetpp

        print('Attention mode: {}'.format(attention_mode))

        self.image_shape = (image_size, image_size)

        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.output_val_images_dir, exist_ok=True)

        if len(annottaion_str) > 0:
            with open(self.annotation_file, 'w') as f:
                f.write(annottaion_str + '\n')

        if full_args is not None:
            save_args_file = os.path.join(experiment_folder, 'args.yaml')
            with open(save_args_file, 'w') as f:
                yaml.safe_dump(vars(full_args), f)

        if train_data_paths is not None:
            self.train_base_dataset = PairedDenoiseDataset(
                    noisy_images_path=train_data_paths[0],
                    clear_images_path=train_data_paths[1],
                    need_crop=True,
                    window_size=self.image_shape[0],
                    optional_dataset_size=80000,
                    preload=preload_data,
                    use_ycrcb=use_ycrcb,
                    grayscale=grayscale
                )
        else:
            self.train_base_dataset = None

        if synth_data_paths is not None:
            self.train_synth_dataset = SyntheticNoiseDataset(
                clear_images_path=synth_data_paths,
                window_size=self.image_shape[0],
                preload=preload_data,
                optional_dataset_size=20000,
                use_ycrcb=use_ycrcb,
                grayscale=grayscale
            )

            if self.train_base_dataset is not None:
                self.train_base_dataset = torch.utils.data.ConcatDataset(
                    [self.train_base_dataset, self.train_synth_dataset]
                )
            else:
                self.train_base_dataset = self.train_synth_dataset

        assert self.train_base_dataset is not None, 'Please set one of datasets: Train or Synthetic'

        self.val_dataset = PairedDenoiseDataset(
            noisy_images_path=val_data_paths[0],
            clear_images_path=val_data_paths[1],
            need_crop=False,
            return_names=True,
            preload=preload_data,
            use_ycrcb=use_ycrcb,
            grayscale=grayscale
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_base_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_workers
        )

        self.images_visualizer = None if visdom_port is None else VisImage(
            title='Denoising',
            port=visdom_port,
            vis_step=150,
            scale=2,
            use_ycrcb=use_ycrcb,
            grayscale=grayscale
        )

        self.attention_visualizer = None if visdom_port is None else VisAttentionMaps(
            title='Denoising',
            port=visdom_port,
            vis_step=150,
            scale=1.5,
            maps_count=4
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

        ch_count = 1 if grayscale else 3

        if use_uformer:
            self.model = Uformer(
                img_size=image_size, embed_dim=32, win_size=8, 
                token_projection='linear', token_mlp='leff',
                depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True,
                dd_in=ch_count, in_chans=ch_count
            )
        else:
            used_architecture = FFTAttentionUNetPlusPlus if use_unetpp else FFTAttentionUNet
            self.model = used_architecture(
                in_ch=ch_count,
                out_ch=ch_count,
                image_size=image_size,
                use_substraction=substracted_noise,
                attention_mode=attention_mode
            )

        self.loss_weighter = ModelMultitask(losses_count=2)
        self.loss_weighter = self.loss_weighter.to(self.device)

        self.model.apply(init_weights)
        self.model = self.model.to(device)

        # self.optimizer = torch.optim.SGD(
        #     params=[{'params': self.model.parameters()}, {'params': self.loss_weighter.parameters(), 'weight_decay': 0}], 
        #     lr=init_lr, nesterov=True, momentum=0.9, weight_decay=1e-4
        # )
        # self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        self.optimizer = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}, {'params': self.loss_weighter.parameters(), 'weight_decay': 0}], 
            lr=init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
        # self.optimizer = torch.optim.RAdam(
        #     params=[{'params': self.model.parameters()}, {'params': self.loss_weighter.parameters(), 'weight_decay': 0}], 
        #     lr=init_lr, betas=(0.9, 0.999), eps=1e-8, decoupled_weight_decay=True, weight_decay=1e-2
        # )
        # self.optimizer = AdaSmooth(
        #     params=[{'params': self.model.parameters()}, {'params': self.loss_weighter.parameters(), 'weight_decay': 0}], 
        #     lr=init_lr, weight_decay=1e-2, weight_decouple=True
        # )

        if load_path is not None:
            load_data = torch.load(load_path, map_location=self.device)

            self.model.load_state_dict(load_data['model'])
            print(
                '#' * 5 + ' Model has been loaded by path: {} '.format(load_path) +  '#' * 5
            )

            if not no_load_optim:
                self.optimizer.load_state_dict(load_data['optimizer'])
                print(
                    '#' * 5 + ' Optimizer has been loaded by path: {} '.format(load_path) + '#' * 5
                )
                self.optimizer.param_groups[0]['lr'] = init_lr
                print('Optimizer LR: {:.5f}'.format(self.get_lr()))

                if hasattr(self, 'loss_weighter') and self.loss_weighter is not None and 'loss_weighter' in load_data.keys():
                    self.loss_weighter.load_state_dict(load_data['loss_weighter'])
                    print('Loss weighter sigmas have been loaded')

            self.optimizer.param_groups[0]['weight_decay'] = 1e-4
            print('Optimizer Weights Decay: {:.5f}'.format(self.optimizer.param_groups[0]['weight_decay']))
            print('Loss Weights Decay: {:.5f}'.format(self.optimizer.param_groups[1]['weight_decay']))

        self.images_criterion = CharbonnierLoss().to(self.device)
        # self.images_criterion = FocalFrequencyLoss(patch_factor=16, loss_weight=10).to(self.device)
        # self.images_criterion = MIXLoss(data_range=1.0, channel=ch_count)
        self.val_criterion = self.images_criterion
        # self.perceptual_loss = DISTS().to(self.device)
        self.perceptual_loss = None
        # self.final_hist_loss = HistLoss(image_size=128, device=self.device)
        self.final_hist_loss = None
        # self.adv_loss = Adversarial(image_size=self.image_shape[0], gan_type='GAN', spectral_norm=True, in_ch=ch_count).to(device)
        self.adv_loss = None
        # self.hf_loss = HightFrequencyFFTLoss(self.image_shape).to(device)
        # self.hf_loss = HFENLoss(
        #     loss_f=CharbonnierLoss().to(self.device),
        #     norm=False
        # )
        # self.edges_loss = LapLoss().to(device)
        # self.tv_loss = TVLoss(tv_loss_weight=0.5)
        # self.fdl_loss = FDL_loss().to(self.device)

        # self.ssim_loss = None
        self.accuracy_measure = TorchPSNR(data_range=1.0).to(device)
        self.ssim_measure = SSIM(data_range=1.0, channel=ch_count)

        self.mixup = MixUp_AUG()

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
        else:
            self.scheduler = None


        self.model = torch.compile(self.model, mode='reduce-overhead')

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_step(self, epoch) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        avg_epoch_loss = 0

        batches_count = len(self.train_dataloader)

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for idx, (_noisy_image, _clear_image) in enumerate(self.train_dataloader):
                # Take YCrCb in 0..1 data range
                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device)
                # clear_image = kornia.enhance.sharpness(clear_image, 2.0)

                if epoch > 3 and np.random.randint(0, 101) > SYNTH_CONFIG['MIXUP']:
                    clear_image, noisy_image = self.mixup.aug(clear_image, noisy_image)

                output = self.model(noisy_image)

                pred_images = output[0]
                spatial_attention_maps = output[1]

                # Pixel-wise loss compuited in 0..1 data range
                loss = calculate_loss(pred_images, clear_image, self.images_criterion, self.use_unetpp)

                p_loss = float(0)
                if self.perceptual_loss is not None:
                    # Perceptual loss calculated in RGB 0..1
                    p_loss = calculate_loss(
                        pred_images,
                        self._convert_to_rgb(clear_image),
                        lambda x, y: self.perceptual_loss(self._convert_to_rgb(x), y),
                        self.use_unetpp
                    )

                # f_loss = calculate_loss(
                #     pred_images,
                #     # kornia.enhance.sharpness(clear_image[:, :1] if self.use_ycrcb or self.grayscale else kornia.color.rgb_to_y(clear_image), 2.0),
                #     clear_image[:, :1] if self.use_ycrcb or self.grayscale else kornia.color.rgb_to_y(clear_image),
                #     lambda x, y: self.hf_loss(
                #         x[:, :1] if self.use_ycrcb or self.grayscale else kornia.color.rgb_to_y(x),
                #         y
                #     ),
                #     self.use_unetpp
                # )

                # tv_loss_value = self.tv_loss(pred_images)

                # a_loss = calculate_loss(pred_images, clear_image, self.adv_loss, self.use_unetpp)

                # e_loss = calculate_loss(pred_images, clear_image, self.edges_loss, self.use_unetpp)

                # f_loss = calculate_loss(
                #     pred_images,
                #     self._convert_to_rgb(clear_image),
                #     lambda x, y: self.hf_loss(self._convert_to_rgb(x), y),
                #     self.use_unetpp
                # )

                # h_loss = calculate_loss(
                #     pred_images,
                #     self._convert_to_rgb(clear_image),
                #     lambda x, y: self.final_hist_loss(self._convert_to_rgb(x), y),
                #     self.use_unetpp
                # )

                # p_loss = calculate_loss(
                #     pred_images,
                #     self._convert_to_rgb(clear_image),
                #     lambda x, y: self.fdl_loss(self._convert_to_rgb(x), y),
                #     self.use_unetpp
                # )

                # total_loss = self.loss_weighter([loss, p_loss])
                total_loss = loss


                if self.gradient_accumulation_steps > 1:
                    total_loss = total_loss / self.gradient_accumulation_steps

                total_loss.backward()

                if (self.gradient_accumulation_steps <= 1) or (
                        (idx + 1) % self.gradient_accumulation_steps == 0) or (
                        idx + 1 == batches_count):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    self.optimizer.zero_grad()

                pbar.postfix = \
                    'Epoch: {}/{}, loss: {:.7f}, w: [{:.2f}, {:.2f}]'.format(
                        epoch,
                        self.epochs,
                        loss.item(),
                        # p_loss.item(),
                        self.loss_weighter.sigma[0].item(),
                        self.loss_weighter.sigma[1].item()
                    )
                avg_epoch_loss += loss.item() / len(self.train_dataloader)

                if self.images_visualizer is not None:
                    with torch.no_grad():
                        vis_idx = self.images_visualizer.per_batch(
                            {
                                'input_img': noisy_image,
                                'pred_image': pred_images[0].detach() if self.use_unetpp else pred_images.detach(),
                                'gt_image': clear_image
                            }
                        )

                        self.attention_visualizer.per_batch(
                            {
                                'sa_list': spatial_attention_maps
                            },
                            i=vis_idx
                        )

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

                assert noisy_image.size(1) >= self.image_shape[0] and noisy_image.size(2) >= self.image_shape[1], \
                    str(noisy_image.shape)

                with torch.no_grad():
                    restored_image = denoise_inference(
                        tensor_img=noisy_image, model=self.model, window_size=self.image_shape[0], 
                        batch_size=self.batch_size, crop_size=0
                    ).unsqueeze(0)

                    loss = self.val_criterion(restored_image, clear_image)
                    
                    avg_loss_rate += loss.item()

                    # rgb_restored_image = self._convert_to_rgb(restored_image)
                    # rgb_clear_image = self._convert_to_rgb(clear_image)

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

                    result_path = os.path.join(self.output_val_images_dir, image_name)
                    val_img = (restored_image.squeeze(0).to('cpu').permute(1, 2, 0) * 255.0).numpy().astype(np.uint8)
                    val_img = val_img[..., 0] if self.grayscale else val_img

                    Image.fromarray(val_img).save(result_path)

        if test_len > 0:
            avg_acc_rate /= test_len
            avg_loss_rate /= test_len
            avg_ssim_rate /= test_len

        if self.scheduler is not None:
            old_lr = self.get_lr()
            self.scheduler.step()
            new_lr = self.get_lr()
            if abs(old_lr - new_lr) > 1E-9:
                print('LR has changed from {:.8f} to {:.8f}'.format(old_lr, new_lr))

        return avg_loss_rate, (avg_acc_rate, avg_ssim_rate)

    def _convert_to_rgb(self, _tensor: torch.Tensor) -> torch.Tensor:
        return convert_tensor_to_rgb(_tensor, self.use_ycrcb, self.grayscale)

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
            'model': self.model._orig_mod.state_dict() 
                        if hasattr(self.model, '_orig_mod') else 
                            self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'acc': avg_acc_rate,
            'epoch': epoch
        }

        if self.adv_loss is not None:
            save_state['gan'] = self.adv_loss.state_dict()

        if hasattr(self, 'loss_weighter') and self.loss_weighter is not None:
            save_state['loss_weighter'] = self.loss_weighter.state_dict()


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
        '--train_data_folder', type=str, required=False,
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
        '--resume_epoch', type=int, required=False, default=1
    )
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
        '--grad_accum_steps', type=int, required=False, default=1,
        help='Count of batches to accumulate gradiets.'
    )
    parser.add_argument(
        '--attention_mode', type=str, required=False, default='full',
        choices=['full', 'ca', 'sa', 'cbam', 'none'],
        help='Attention mode from \'full\', \'ca\', \'sa\', \'cbam\', \'none\'.'
    )
    parser.add_argument(
        '--use_unetplusplus', action='store_true',
        help='Use U-Net++ architecture.'
    )
    parser.add_argument(
        '--use_uformer', action='store_true',
        help='Use U-Former architecture.'
    )
    parser.add_argument(
        '--use_ycrcb', action='store_true',
        help='Use YCrCb color space for image training.'
    )
    parser.add_argument(
        '--use_grayscale', action='store_true',
        help='Use 1-channel for image training.'
    )
    parser.add_argument(
        '--substracted_noise', action='store_true',
        help='Use netwotk prediction as Y = X + F(X).'
    )
    parser.add_argument(
        '--batch_size', type=int, required=False, default=32,
        help='Training batch size.'
    )
    parser.add_argument(
        '--lr', type=float, required=False, default=0.001,
        help='Start value of learning rate.'
    )
    parser.add_argument(
        '--lr_milestones', type=int, required=False, default=3,
        help='Count or learning rate scheduler milestones.'
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
    # torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')

    args = parse_args()

    if args.train_data_folder is not None:
        train_data = (
            os.path.join(args.train_data_folder, 'noisy/'),
            os.path.join(args.train_data_folder, 'clear/')
        )
    else:
        train_data = None

    val_data = (
        os.path.join(args.validation_data_folder, 'noisy/'),
        os.path.join(args.validation_data_folder, 'clear/')
    )

    CustomTrainingPipeline(
        train_data_paths=train_data,
        val_data_paths=val_data,
        synth_data_paths=args.synthetic_data_paths,
        experiment_folder=args.experiment_folder,
        load_path=args.load_path,
        visdom_port=args.visdom_port,
        epochs=args.epochs,
        resume_epoch=args.resume_epoch,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_workers=args.njobs,
        preload_data=args.preload_datasets,
        init_lr=args.lr,
        annottaion_str=args.annotation,
        lr_steps=args.lr_milestones,
        no_load_optim=args.no_load_optim,
        gradient_accumulation_steps=args.grad_accum_steps,
        use_ycrcb=args.use_ycrcb,
        attention_mode=args.attention_mode,
        grayscale=args.use_grayscale,
        use_unetpp=args.use_unetplusplus,
        use_uformer=args.use_uformer,
        substracted_noise=args.substracted_noise,
        full_args=args
    ).fit()

    exit(0)
