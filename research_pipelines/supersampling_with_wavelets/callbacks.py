import os
import torch
import random
import numpy as np
from functools import reduce
from visdom import Visdom
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F


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


class VisImagesGrid(AbstractCallback):
    def __init__(self, title, server='http://localhost', port=8080,
                 vis_step=1, scale=10, grid_size=8):
        self.viz = Visdom(server=server, port=port)
        self.title = title + 'Image'
        self.windows = {1: None}
        self.n = 0
        self.step = vis_step
        self.scale = scale

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)

        self.to_image = ToPILImage()
        self.to_tensor = ToTensor()

        self.grid_size = grid_size

        random.seed()

    def per_batch(self, args, label=1):
        """
        Per batch visualization
        Args:
            args: input tensor in [0, 1] values format
            label: 1

        Returns:

        """
        if self.n % self.step == 0:
            # i = random.randint(0, args['img'].size(0) - 1)
            # i = (args['img'].size(0) - 1) // 2

            for win in self.windows.keys():
                if win == label:
                    _grid_size = int(np.sqrt(args['img'].size(0)))

                    grid_size = _grid_size \
                        if _grid_size < self.grid_size else self.grid_size

                    imgs = args['img'].to('cpu')

                    grid_lines = []
                    for line in range(grid_size):
                        grid_lines.append(
                            torch.cat(
                                tuple(imgs[line*grid_size:(line+1)*grid_size]),
                                dim=2
                            )
                        )

                    grid = torch.cat(tuple(grid_lines), dim=1)
                    grid = grid * self.std + self.mean
                    grid = torch.clamp(grid, 0, 1)

                    self.windows[win] = self.viz.image(
                        F.interpolate(
                            grid.unsqueeze(0),
                            scale_factor=(self.scale, self.scale)
                        ).squeeze(0),
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