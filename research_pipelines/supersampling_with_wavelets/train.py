from argparse import ArgumentParser, Namespace
from typing import Tuple
import tqdm
import torch
from torch.utils import data
import torchvision
import timm
import os

from dataloader import ClassificationDataloader
from callbacks import VisImagesGrid, VisPlot


class CustomTrainingPipeline(object):
    def __init__(self,
                 data_path: str,
                 experiment_folder: str,
                 model_name: str = 'resnet18',
                 load_path: str = None,
                 visdom_port: int = 9000,
                 batch_size: int = 32,
                 epochs: int = 200,
                 resume_epoch: int = 1,
                 stop_criteria: float = 1E-7,
                 device: str = 'cuda',
                 image_size: int = 224):
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
        self.dataset_path = data_path
        train_data_path = os.path.join(data_path, 'train/')
        val_data_path = os.path.join(data_path, 'val/')
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

        self.train_dataset = ClassificationDataloader(
            train_data_path,
            self.image_shape,
            True,
            os.path.join(experiment_folder, 'train_data.csv'),
            os.path.join(experiment_folder, 'classes.txt')
        )
        self.val_dataset = ClassificationDataloader(
            val_data_path,
            self.image_shape,
            False,
            os.path.join(experiment_folder, 'val_data.csv'),
            os.path.join(experiment_folder, 'classes.txt')
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.train_dataset.get_classes_sampler(),
            drop_last=True,
            num_workers=4
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        self.batch_visualizer = None if visdom_port is None else VisImagesGrid(
            title='Classification',
            port=visdom_port,
            vis_step=250,
            scale=1,
            grid_size=8
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
                ylabel='Accuracy',
                legend=['val']
            )

        self.model = timm.create_model(model_name, num_classes=self.train_dataset.num_classes)

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(load_path, map_location='cpu'))
            print(
                'Model has been loaded by path: {}'.format(load_path)
            )
        self.model = self.model.to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RAdam(
            params=self.model.parameters(), lr=0.01)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, verbose=True, factor=0.1
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[15, 30, 50],
            gamma=0.1
        )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _train_step(self, epoch) -> float:
        self.model.train()
        avg_epoch_loss = 0

        with tqdm.tqdm(total=len(self.train_dataloader)) as pbar:
            for i, (_x, _y) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x = _x.to(self.device)
                y_truth = _y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_truth)

                loss.backward()
                self.optimizer.step()

                pbar.postfix = \
                    'Epoch: {}/{}, loss: {:.8f}'.format(
                        epoch,
                        self.epochs,
                        loss.item() / self.train_dataloader.batch_size
                    )
                avg_epoch_loss += \
                    loss.item() / len(self.train_dataloader)

                if self.batch_visualizer is not None:
                    self.batch_visualizer.per_batch(
                        {
                            'img': x
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
            for _x, _y in tqdm.tqdm(self.val_dataloader):
                x = _x.to(self.device)
                y_truth = _y.to(self.device)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_truth)
                avg_loss_rate += loss.item()
                acc_rate = torch.eq(y_truth, y_pred.argmax(
                    dim=1)).sum() / y_truth.size(0)
                acc_rate = float(acc_rate.to('cpu').numpy())

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
        self._save_best_traced_model(latest_traced_model_path)
        self.model = self.model.to(self.device)

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
            self._save_best_traced_model(best_traced_model_path)
            self.model = self.model.to(self.device)

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
        '--data', type=str, required=True,
        help='Path to training data root folder.'
    )
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

    CustomTrainingPipeline(
        data_path=args.data,
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