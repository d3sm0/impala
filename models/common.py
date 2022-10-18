from typing import Tuple

import numpy as np
import torch
import torch.nn.init
from torch import nn as nn


# TODO: this architecture  is not quite right. Missing layer norm
class ResidualBlock(nn.Module):

    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=3,
                          padding=1)
        self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale)
        conv1 = nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=3,
                          padding=1)
        self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        conv = nn.Conv2d(in_channels=self._input_shape[0],
                         out_channels=self._out_channels,
                         kernel_size=3,
                         padding=1)
        self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0)
        scale = scale / np.sqrt(2)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNNLarge(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...]):
        super(ImpalaCNNLarge, self).__init__()
        c, h, w = input_shape
        shape = (c, h, w)
        conv_seqs = []
        chans = [16, 32, 32]
        # Not fully sure about the logic behind this but its used in PPG code
        scale = 1 / np.sqrt(len(chans))
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        conv_seqs += [
            nn.ReLU(),
            nn.Flatten(),
        ]
        self._output_dim = np.prod(shape)
        self.body = nn.Sequential(*conv_seqs)

    def forward(self, x):
        return self.body(x)

    @property
    def output_dim(self):
        return self._output_dim


class ImpalaCNNSmall(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...]):
        super(ImpalaCNNSmall, self).__init__()
        c, h, w = input_shape
        self.body = nn.Sequential(
            layer_init_normed(nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4), norm_dim=(1, 2, 3),
                              scale=1),
            nn.ReLU(),
            layer_init_normed(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2), norm_dim=(1, 2, 3),
                              scale=1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.body(x)


# TODO: fix initalization here
class AtariBody(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...]):
        super().__init__()
        self.body = nn.Sequential(
            layer_init_truncated(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init_truncated(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init_truncated(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten())

    def forward(self, x):
        return self.body(x)

    @property
    def output_dim(self):
        return 7 * 7 * 64


class LinearBody(nn.Module):
    def __init__(self, observation_space, d_model=256):
        super(LinearBody, self).__init__()
        self.body = nn.Sequential(
            layer_init_ortho(nn.Linear(observation_space, d_model)),
            nn.Tanh(),
            layer_init_ortho(nn.Linear(d_model, d_model)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.body(x)


def layer_init_truncated(layer, bias_const=0.0):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d):
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer)
        else:
            # the weight matrix is [out, in]
            fan_in = layer.weight.shape[1]
        std = 1.0 / np.sqrt(fan_in)
        torch.nn.init.trunc_normal_(layer.weight, std=std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_ortho(layer, std=np.sqrt(2), bias_const=0.0):
    with torch.no_grad():
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim: Tuple = (1,), scale=0.1):
    with torch.no_grad():
        norm = layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.weight.data.mul_(scale / norm)
        torch.nn.init.constant_(layer.bias, 0)
    return layer
