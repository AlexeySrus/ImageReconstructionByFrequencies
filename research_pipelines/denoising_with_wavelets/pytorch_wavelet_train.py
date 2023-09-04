import logging
from argparse import ArgumentParser, Namespace
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

from dataloader import SeriesAndComputingClearDataset
from callbacks import VisImage, VisPlot
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from IS_Net.isnet import ISNetDIS
from IS_Net.discriminator import Discriminator


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class CustomTrainingPipeline(object):
    def __init__(self,
                 train_data_paths: Tuple[str, str],
                 val_data_paths: Tuple[str, str],
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

        self.train_dataset = SeriesAndComputingClearDataset(
            images_series_folder=train_data_paths[0],
            clear_images_path=train_data_paths[1],
            window_size=self.image_shape[0],
            dataset_size=10000
        )
        self.val_dataset = SeriesAndComputingClearDataset(
            images_series_folder=val_data_paths[0],
            clear_images_path=val_data_paths[1],
            window_size=self.image_shape[0],
            dataset_size=128
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=train_workers
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
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
        self.discriminator = Discriminator()

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(load_path, map_location='cpu'))
            print(
                'Model has been loaded by path: {}'.format(load_path)
            )
        self.model = self.model.to(device)
        self.discriminator = self.discriminator.to(device)

        self.images_criterion = torch.nn.MSELoss(size_average=True)
        self.ssim_loss = SSIMLoss()
        self.wavelets_criterion = torch.nn.SmoothL1Loss(size_average=True)
        self.accuracy_measure = TorchPSNR().to(device)

        self.optimizer = torch.optim.RAdam(
            params=self.model.parameters(), lr=0.01)
        self.discriminator_optimizer = torch.optim.RAdam(
            params=self.discriminator.parameters(), lr=0.01
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[150, 300, 5000],
            gamma=0.1
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _compute_wavelets_loss(self, pred_wavelets_pyramid, gt_image, factor: float = 0.9):
        gt_d0_ll = gt_image
        _loss = None
        _loss_scale = 1.0

        for i in range(len(pred_wavelets_pyramid)):
            gt_ll, gt_lh, gt_hl, gt_hh = self.dwt(gt_d0_ll)
            gt_wavelets = torch.cat((gt_ll, gt_lh, gt_hl, gt_hh), dim=1)

            if i == 0:
                _loss = self.wavelets_criterion(pred_wavelets_pyramid[i], gt_wavelets) * _loss_scale * 0.5
            else:
                _loss += self.wavelets_criterion(pred_wavelets_pyramid[i], gt_wavelets) * _loss_scale

            _loss_scale *= factor

            gt_d0_ll = gt_ll / 2.0

        return _loss

    def _train_step(self, epoch) -> float:
        self.model.train()
        self.discriminator.train()
        avg_epoch_loss = 0

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (_noisy_image, _clear_image) in enumerate(self.train_dataloader):
                real_label = 1.
                fake_label = 0.

                noisy_image = _noisy_image.to(self.device)
                clear_image = _clear_image.to(self.device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                label = torch.full((clear_image.size(0),), real_label, dtype=torch.float, device=self.device)

                self.discriminator_optimizer.zero_grad()

                real_d_out = self.discriminator(clear_image)
                errD_real = torch.nn.functional.binary_cross_entropy(real_d_out, label)
                errD_real.backward()
                D_x = real_d_out.mean().item()

                pred_wavelets_pyramid, _ = self.model(noisy_image)
                pred_ll, pred_lh, pred_hl, pred_hh = torch.split(pred_wavelets_pyramid[0], 3, dim=1)
                restored_image = self.iwt(pred_ll, pred_lh, pred_hl, pred_hh)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(restored_image.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = torch.nn.functional.binary_cross_entropy(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.discriminator_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizer.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(restored_image)
                # Calculate G's loss based on this output
                errG = torch.nn.functional.binary_cross_entropy(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer.step()

                ############################
                # (3) Additional: Pixel-wise and wavelets losses
                ###########################
                # pred_wavelets_pyramid, _ = self.model(noisy_image)
                # pred_ll, pred_lh, pred_hl, pred_hh = torch.split(pred_wavelets_pyramid[0], 3, dim=1)
                #
                # restored_image = self.iwt(pred_ll, pred_lh, pred_hl, pred_hh)
                #
                # loss = self.images_criterion(restored_image, clear_image)   # / 2 + self.ssim_loss(restored_image, clear_image) / 2
                # wloss = self._compute_wavelets_loss(pred_wavelets_pyramid, clear_image)
                #
                # total_loss = loss + wloss
                #
                # total_loss.backward()
                # self.optimizer.step()

                pbar.postfix = \
                    'Epoch: {}/{}, D_loss: {:.8f}, G_loss: {:.8f}'.format(
                        epoch,
                        self.epochs,
                        errD.item(),
                        errG.item()
                    )
                avg_epoch_loss += \
                    (errD + errG).item() / len(self.train_dataloader)

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

        if self.val_dataloader is not None:
            for _noisy_image, _clear_image in tqdm.tqdm(self.val_dataloader):
                with torch.no_grad():
                    noisy_image = _noisy_image.to(self.device)
                    clear_image = _clear_image.to(self.device)

                    pred_wavelets_pyramid, _ = self.model(noisy_image)

                    restored_image = self.iwt(*torch.split(pred_wavelets_pyramid[0], 3, dim=1))

                    loss = self.images_criterion(restored_image, clear_image)   #  / 2 + self.ssim_loss(restored_image, clear_image) / 2
                    loss += self._compute_wavelets_loss(pred_wavelets_pyramid, clear_image)
                    avg_loss_rate += loss.item()

                    val_psnr = self.accuracy_measure(
                        restored_image,
                        clear_image
                    )

                    acc_rate = val_psnr.item()   # float(acc_rate.to('cpu').numpy())

                    avg_acc_rate += acc_rate
                    test_len += 1

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
        model_save_path = os.path.join(
            self.checkpoints_dir,
            'resnet_epoch_{}_acc_{:.2f}.trh'.format(epoch, avg_acc_rate)
        )
        best_model_path = os.path.join(
            self.checkpoints_dir,
            'best.trh'
        )
        best_discriminator_path = os.path.join(
            self.checkpoints_dir,
            'best_discriminator.trh'
        )
        best_traced_model_path = os.path.join(
            self.experiment_folder,
            'traced_best_model.pt'
        )
        latest_traced_model_path = os.path.join(
            self.experiment_folder,
            'traced_latest_model.pt'
        )

        self.model = self.model.to('cpu')
        self.model.eval()
        self.discriminator.eval()
        # self._save_best_traced_model(latest_traced_model_path)
        self.model = self.model.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        if self.best_test_score - avg_acc_rate < -1E-5:
            self.best_test_score = avg_acc_rate

            self.model = self.model.to('cpu')
            self.model.eval()
            torch.save(
                self.model.state_dict(),
                model_save_path
            )
            torch.save(
                self.model.state_dict(),
                best_model_path
            )
            torch.save(
                self.discriminator.state_dict(),
                best_discriminator_path
            )
            # self._save_best_traced_model(best_traced_model_path)
            self.model = self.model.to(self.device)
            self.discriminator = self.discriminator.to(self.device)

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
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/images_series/',
        '/media/alexey/SSDData/datasets/denoising_dataset/real_sense_noise_val/averaged_outclass_series/'
    )

    CustomTrainingPipeline(
        train_data_paths=train_data,
        val_data_paths=val_data,
        experiment_folder=args.experiment_folder,
        load_path=args.load_path,
        visdom_port=args.visdom_port,
        epochs=args.epochs,
        resume_epoch=args.resume_epoch,
        batch_size=args.batch_size,
        model_name=args.model,
        image_size=args.image_size
    ).fit()

    exit(0)
