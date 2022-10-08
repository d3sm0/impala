from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim: Tuple = (1,), scale=0.1):
    with torch.no_grad():
        norm = layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.weight.data.mul_(scale / norm)
        torch.nn.init.constant_(layer.bias, 0)
    return layer


class LinearBody(nn.Module):
    def __init__(self, observation_space, d_model=256):
        super(LinearBody, self).__init__()
        self.body = nn.Sequential(
            layer_init(nn.Linear(observation_space, d_model)),
            nn.Tanh(),
            layer_init(nn.Linear(d_model, d_model)),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.body(x)


class ImpalaActorCritic(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...], action_dim: int, h_dim=100):
        super().__init__()
        self.body = ImpalaCNN(obs_dim, d_model=h_dim)
        self.policy = layer_init_normed(nn.Linear(h_dim, action_dim))
        self.value = nn.Sequential(layer_init_normed(nn.Linear(h_dim, 1)), nn.Flatten(start_dim=-2))

    def forward(self, s):
        h = self.body(s / 255.)
        pi = self.policy(h)
        v = self.value(h)
        return pi, v


class AtariActorCritic(nn.Module):
    def __init__(self, observation_space=None, action_space=None):
        super(AtariActorCritic, self).__init__()
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, 6), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        x = x / 255.
        h = self.body(x)
        return self.actor(h), self.critic(h)


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, d_model=64):
        super(ActorCritic, self).__init__()
        observation_space, = observation_space
        self.actor = nn.Sequential(LinearBody(observation_space, d_model),
                                   layer_init(nn.Linear(d_model, action_space, bias=True), std=0.01))
        self.critic = nn.Sequential(LinearBody(observation_space, d_model), layer_init(nn.Linear(d_model, 1), std=1.),
                                    nn.Flatten(start_dim=-2))

    def forward(self, x):
        return self.actor(x), self.critic(x)


class ActorCriticQ(nn.Module):

    def __init__(self, observation_space, action_space, d_model=64):
        super(ActorCriticQ, self).__init__()
        observation_space, = observation_space
        self.actor = nn.Sequential(LinearBody(observation_space, d_model),
                                   layer_init(nn.Linear(d_model, action_space, bias=True), std=0.01))
        self.q = nn.Sequential(LinearBody(observation_space, d_model),
                               layer_init(nn.Linear(d_model, action_space), std=1.))

        self.critic = nn.Sequential(LinearBody(observation_space, d_model), layer_init(nn.Linear(d_model, 1), std=1.),
                                    nn.Flatten(start_dim=-2))

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def q_value(self, x):
        return self.q(x)


class VFunction(nn.Module):
    def __init__(self, observation_space, d_model=100):
        super().__init__()

        self.body = ImpalaCNN(observation_space, d_model=d_model)
        self.critic = layer_init_normed(nn.Linear(in_features=d_model, out_features=1))

    def forward(self, s):
        n, t = s.shape[:2]
        s = s.flatten(0, 1)
        return self.critic(self.body(s)).reshape(n, t)


class Agent(nn.Module):
    """A simple network."""

    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 32):
        super().__init__()
        self._num_actions = action_dim
        self.actor = ImpalaActorCritic(obs_dim, action_dim, h_dim)
        self.critic = VFunction(obs_dim, h_dim)

    def forward(self, observation) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of observations."""
        hidden = observation
        policy = self.actor(hidden)
        baseline = self.critic(hidden)
        return policy, baseline


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


class ImpalaCNN(nn.Module):
    def __init__(self, input_shape, d_model: int = 256):
        super(ImpalaCNN, self).__init__()
        h, w, c = input_shape
        shape = (c, h, w)
        conv_seqs = []
        chans = [16, 32, 32]
        # Not fully sure about the logic behind this but its used in PPG code
        scale = 1 / np.sqrt(len(chans))
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        head = nn.Linear(in_features=int(np.prod(shape)), out_features=d_model)
        head = layer_init_normed(head, scale=1.4)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            head,
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, x):
        return self.network(x)


_register = {"procgen": ImpalaActorCritic, "atari": AtariActorCritic, "control": ActorCritic}


def get_model(task_id):
    return _register[task_id]
