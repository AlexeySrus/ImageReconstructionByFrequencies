from typing import Optional
import os
import cv2
import torch
import random
import numpy as np
from functools import reduce
from visdom import Visdom
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F

from utils.image_utils import merge_by_wavelets
from utils.tensor_utils import preprocess_image


def add_prefix(path, pref):
    """
    Add prefix to file in path
    Args:
        path: path to file
        pref: prefixvectors2line
    Returns:
        path to file with named with prefix
    """
    splitted_path = list(os.path.split(path))
    splitted_path[-1] = pref + splitted_path[-1]
    return reduce(lambda x, y: x + '/' + y, splitted_path)


class AbstractCallback(object):
    def per_batch(self, args):
        raise RuntimeError("Don\'t implement batch callback method")

    def per_epoch(self, args):
        raise RuntimeError("Don\'t implement epoch callback method")

    def early_stopping(self, args):
        raise RuntimeError("Don\'t implement early stopping callback method")


class SaveBestModelAndOptimizer(AbstractCallback):
    def __init__(self,
                 path: str,
                 metric_mode: str = 'max',
                 target_metric: str = 'val acc'):
        self.path = path
        self.best_metric = 1000 if metric_mode == 'min' else -1000
        self.metric_mode = metric_mode
        self.target_metric = target_metric

        if not os.path.isdir(path):
            os.makedirs(path)

        self.file_with_best_metric = os.path.join(
            path,
            '_best_measure.txt'
        )

        if os.path.isfile(self.file_with_best_metric):
            with open(self.file_with_best_metric, 'r') as mf:
                self.best_metric = float(mf.read())

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        target_metric_value = args[self.target_metric]

        if self.metric_mode == 'min':
            is_save_new = target_metric_value - self.best_metric < 1E-5
        else:
            is_save_new = target_metric_value - self.best_metric > -1E-5

        if is_save_new:
            self.best_metric = target_metric_value

            args['model'].save(
                os.path.join(self.path, 'best_model.pt')
            )

            torch.save(args['optimize_state'], (
                os.path.join(
                    self.path,
                    'best_optimize_state.pt'
                )
            ))

            with open(self.file_with_best_metric, 'w') as mf:
                mf.write(str(float(self.best_metric)))

    def early_stopping(self, args):
        pass


class SaveLatestModelAndOptimizer(AbstractCallback):
    def __init__(self,
                 path: str):
        self.path = path

        if not os.path.isdir(path):
            os.makedirs(path)

    def per_batch(self, args):
        pass

    def per_epoch(self, args):
        args['model'].save(
            os.path.join(self.path, 'latest_model.trh')
        )

        torch.save(args['optimize_state'], (
            os.path.join(
                self.path,
                'latest_optimize_state.trh'
            )
        ))

    def early_stopping(self, args):
        pass


class VisPlot(AbstractCallback):
    def __init__(self, title, server='http://localhost', port=8080,
                 logname=None):
        self.viz = Visdom(server=server, port=port, log_to_filename=logname)
        self.windows = {}
        self.title = title

    def register_scatterplot(self, name, xlabel, ylabel, legend=None):
        options = dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel) if legend is None \
                       else dict(title=self.title, markersize=5,
                        xlabel=xlabel, ylabel=ylabel,
                        legend=legend)

        self.windows[name] = [None, options]

    def update_scatterplot(self, name, x, y1, y2=None, window_size=100):
        """
        Update plot
        Args:
            name: name of updating plot
            x: x values for plotting
            y1: y values for plotting
            y2: plot can contains two graphs
            window_size: window size for plot smoothing (by mean in window)
        Returns:
        """
        if y2 is None:
            self.windows[name][0] = self.viz.line(
                np.array([y1], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )
        else:
            self.windows[name][0] = self.viz.line(
                np.array([[y1, y2]], dtype=np.float32),
                np.array([x], dtype=np.float32),
                win=self.windows[name][0],
                opts=self.windows[name][1],
                update='append' if self.windows[name][0] is not None else None
            )

    def per_batch(self, args, keyward='per_batch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win and 'acc' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss']
                    )

                if 'train' in win and 'acc' in win and 'loss' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['acc']
                    )

                if 'train' in win and 'acc' in win and 'loss' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss'],
                        args['acc']
                    )

    def per_epoch(self, args, keyward='per_epoch'):
        for win in self.windows.keys():
            if keyward in win:
                if 'train' in win and 'validation' in win and 'acc' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['loss'],
                        args['val loss']
                    )

                if 'train' in win and 'validation' in win and 'acc' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['acc'],
                        args['val acc']
                    )

                if 'validation' in win and 'roc' in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['val roc']
                    )

                if 'validation' in win and 'acc' in win and 'train' not in win:
                    self.update_scatterplot(
                        win,
                        args['n'],
                        args['val acc'],
                    )

    def early_stopping(self, args):
        pass


class VisImageForFourier(AbstractCallback):
    def __init__(self, title, server='http://localhost', port=8080,
                 vis_step=1, scale=10, use_ycrcb: bool = False, grayscale: bool = False):
        self.viz = Visdom(server=server, port=port)
        self.title = title + ' input | predicted | original'
        self.windows = {1: None}
        self.n = 0
        self.step = vis_step
        self.scale = scale

        self.img_mean = 0
        self.img_std = 1
        self.w_mean = 0
        self.w_std = 1

        self.use_ycrcb = use_ycrcb
        self.grayscale = grayscale

        random.seed()

    def _denorm_image(self, im: torch.Tensor) -> torch.Tensor:
        return im * self.img_std + self.img_mean

    def _tensor_to_image(self, im: torch.Tensor, cv_out: bool = True) -> np.ndarray:
        _image = self._denorm_image(im)
        out = (_image.permute(1, 2, 0) * 255.0).to('cpu').numpy()
        if cv_out:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    def _to_rgb(self, im: torch.Tensor) -> torch.Tensor:
        np_image = self._tensor_to_image(im)
        if self.use_ycrcb and not self.grayscale:
            rgb_image = cv2.cvtColor(np_image, cv2.COLOR_YCrCb2RGB)
        elif self.grayscale:
            rgb_image = cv2.cvtColor(np_image[..., 0], cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = np_image
        return preprocess_image(rgb_image, self.img_mean, self.img_std)

    def per_batch(self, args, label=1):
        if self.n % self.step == 0:
            i = random.randint(0, args['lr_img'].size(0) - 1)

            for win in self.windows.keys():
                if win == label:
                    # """
                    # 'lr_img': lr_image,
                    # 'gt_image': _hr_image
                    # """
                    lr_img = args['lr_img'][i]
                    gt_wavelets = args['gt_wavelets'][i]
                    pred_wavelets = args['pred_wavelets'][i]
                    gt_image = args['gt_image'][i]

                    input_image = self._denorm_image(self._to_rgb(self._merge_by_wavelets(lr_img, gt_wavelets)))
                    pred_image = self._denorm_image(self._to_rgb(self._merge_by_wavelets(lr_img, pred_wavelets)))
                    gt_image = self._denorm_image(self._to_rgb(gt_image))

                    x = torch.cat(
                        (input_image, pred_image, gt_image),
                        dim=2
                    )

                    self.windows[win] = self.viz.image(
                        F.interpolate(
                            x.unsqueeze(0),
                            scale_factor=(self.scale, self.scale)
                        ).squeeze(),
                        win=self.windows[win],
                        opts=dict(title=self.title)
                    )

        self.n += 1
        if self.n >= 1000000000:
            self.n = 0

    def per_epoch(self, args):
        pass

    def early_stopping(self, args):
        pass

    def add_window(self, label):
        self.windows[label] = None


class VisImage(VisImageForFourier):
    def __init__(self, title, server='http://localhost', port=8080,
                 vis_step=1, scale=10,
                 use_ycrcb: bool = False, grayscale: bool = False):
        super().__init__(title, server, port, vis_step, scale, use_ycrcb, grayscale)

    def per_batch(self, args, label=1, i: Optional[int] = None) -> Optional[int]:
        if self.n % self.step == 0:
            if i is None:
                i = random.randint(0, args['gt_image'].size(0) - 1)

            for win in self.windows.keys():
                if win == label:
                    pred_image = args['pred_image'][i]
                    gt_image = args['gt_image'][i]
                    inp_image = args['input_img'][i]

                    inp_image = torch.clip(inp_image, 0, 1).to('cpu')
                    gt_image = torch.clip(gt_image, 0, 1).to('cpu')
                    pred_image = torch.clip(pred_image, 0, 1).to('cpu')

                    inp_image = self._denorm_image(self._to_rgb(inp_image))
                    gt_image = self._denorm_image(self._to_rgb(gt_image))
                    pred_image = self._denorm_image(self._to_rgb(pred_image))

                    x = torch.cat(
                        (inp_image, pred_image, gt_image),
                        dim=2
                    )

                    self.windows[win] = self.viz.image(
                        F.interpolate(
                            x.unsqueeze(0),
                            scale_factor=(self.scale, self.scale),
                            mode='area'
                        ).squeeze(),
                        win=self.windows[win],
                        opts=dict(title=self.title)
                    )

        self.n += 1
        if self.n >= 1000000000:
            self.n = 0

        return i

    def per_epoch(self, args):
        pass

    def early_stopping(self, args):
        pass

    def add_window(self, label):
        self.windows[label] = None


class VisAttentionMaps(AbstractCallback):
    def __init__(self, title, server='http://localhost', port=8080,
                 vis_step=1, scale=10, maps_count: int = 6):
        self.viz = Visdom(server=server, port=port)
        self.title = title
        for i in range(maps_count):
            self.title += ' | SA {}'.format(i + 1)
        self.windows = {1: None}
        self.n = 0
        self.step = vis_step
        self.scale = scale

        random.seed()

    def per_batch(self, args, label=1, i: Optional[int] = None) -> Optional[int]:
        if self.n % self.step == 0:
            if i is None:
                i = random.randint(0, args['sa_list'][0].size(0) - 1)

            for win in self.windows.keys():
                if win == label:
                    # """
                    # 'sa_list': sa_list
                    # """
                    sa_list = [sa[i] for sa in args['sa_list']]

                    sa_tensors = []
                    step_k = 6
                    for k in range(len(sa_list) // step_k):
                        sa_tensors.append(torch.concat([sa_list[step_k*k + q] for q in range(step_k)], dim=2))

                    sa_tensor = torch.concat(sa_tensors, dim=1)
                    x = sa_tensor
                    
                    # x = torch.cat(
                    #     sa_list,
                    #     dim=2
                    # )
                    x = x.repeat(3, 1, 1)
                    x = torch.clamp(x, 0, 1)

                    self.windows[win] = self.viz.image(
                        F.interpolate(
                            x.unsqueeze(0),
                            scale_factor=(self.scale, self.scale)
                        ).squeeze(),
                        win=self.windows[win],
                        opts=dict(title=self.title)
                    )

        self.n += 1
        if self.n >= 1000000000:
            self.n = 0

        return i

    def per_epoch(self, args):
        pass

    def early_stopping(self, args):
        pass

    def add_window(self, label):
        self.windows[label] = None
