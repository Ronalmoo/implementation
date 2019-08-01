import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """Flatten class"""
    def forward(self, x):
        return torch.flatten(x.size(0), -1)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self._projection = (lambda x, y: x != y)(in_channels, out_channels)
        self._ops = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(3, 1, 1)),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, kernel_size=(3, 1, 1)),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU())
        self._bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        fmap = self._ops(x)
        fmap = F.relu(self._bn(fmap))

        return fmap

