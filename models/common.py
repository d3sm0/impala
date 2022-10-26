from typing import Tuple

import numpy as np
import torch
import torch.nn.init
from torch import nn as nn


# TODO: this architecture  is not quite right. Missing layer norm
# or apply torch.nn.init.spectral_norm
class ResidualBlock(nn.Module):

    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        # scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=3,
                          padding=1)
        self.conv0 = conv0
        conv1 = nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=3,
                          padding=1)
        self.conv1 = conv1

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ImpalaCNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self._out_channels = out_channels
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=self._out_channels,
                         kernel_size=3,
                         padding=1)
        self.conv = conv
        scale = scale / np.sqrt(2)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        # assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNNLarge(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...]):
        super(ImpalaCNNLarge, self).__init__()
        c, h, w = input_shape

        self.body = nn.Sequential(
            ImpalaCNNBlock(c, 16, scale=1.0),
            ImpalaCNNBlock(16, 32, scale=1.0),
            ImpalaCNNBlock(32, 32, scale=1.0),
            # nn.AdaptiveMaxPool2d((8, 8)),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.body(x)

    @property
    def output_dim(self):
        return 2048


class ImpalaCNNSmall(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...]):
        super(ImpalaCNNSmall, self).__init__()
        c, h, w = input_shape
        self.body = nn.Sequential(
            layer_init_truncated(nn.Conv2d(in_channels=c, out_channels=16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init_truncated(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)),
            # nn.AdaptiveMaxPool2d((6, 6)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self._output_dim = 32 * 9 * 9

    def forward(self, x):
        return self.body(x)

    @property
    def output_dim(self):
        return self._output_dim


class AtariBody(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...]):
        super().__init__()
        c, h, w = obs_dim
        self.body = nn.Sequential(
            layer_init_truncated(nn.Conv2d(c, 32, 8, stride=4)),
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
        return 1024


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


def layer_fan_in(layer):
    if isinstance(layer, nn.Conv2d):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
    else:
        fan_in = layer.weight.shape[1]
    return max(1, fan_in)


def layer_init_truncated(layer, scale=1.):
    with torch.no_grad():
        fan_in = layer_fan_in(layer)
        distribution_stddev = np.asarray(.87962566103423978, dtype=np.float32)
        std = np.sqrt(scale / fan_in) / distribution_stddev
        torch.nn.init.trunc_normal_(layer.weight, std=std)
        torch.nn.init.constant_(layer.bias, 0.)
    return layer


def layer_init_uniform(layer, scale: float = 0.33):
    with torch.no_grad():
        fan_in = layer_fan_in(layer)
        scale = np.sqrt(3 / fan_in) * scale
        torch.nn.init.uniform_(layer.weight, -scale, scale)
        torch.nn.init.constant_(layer.bias, 0.)
    return layer


def variance_scaling_uniform(layer, scale: float = 1.):
    with torch.no_grad():
        fan_in = layer_fan_in(layer)
        scale = np.sqrt(3 * scale / fan_in)
        torch.nn.init.uniform_(layer.weight, -scale, scale)
        torch.nn.init.constant_(layer.bias, 0.)
    return layer


def layer_init_ortho(layer, gain=2., bias_const=0.0):
    # defaulted to relu
    with torch.no_grad():
        torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(gain))
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim: Tuple = (1, 2, 3), gain=1):
    with torch.no_grad():
        gain = np.sqrt(gain)
        norm = layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.weight.data.mul_(gain / norm)
        torch.nn.init.constant_(layer.bias, 0)
    return layer
