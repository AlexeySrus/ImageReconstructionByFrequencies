from torch import nn
import torch


class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=20, num_features=64, sum_with_input=True):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)
        self.sum_with_input = sum_with_input

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        if self.sum_with_input:
            out = y + residual
        else:
            out = residual
        return out, [torch.abs(a).unsqueeze(1) for a in [residual[:, 0], residual[:, 1], residual[:, 2], residual.mean(dim=1)]]
