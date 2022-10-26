from typing import Tuple

import torch
import torch.nn as nn

import models.common
import models.quantile_layers


class DistributionalDQN(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...], action_dim: int, h_dim: int = 120, tau_samples: int = 8):
        super().__init__()
        obs_dim, = obs_dim
        self.body = nn.Sequential(
            models.common.layer_init_truncated(nn.Linear(obs_dim, h_dim)),
            nn.ReLU())

        self.q = models.quantile_layers.ImplicitQuantileHead(h_dim, action_dim, d_model=84)
        self.tau_samples = tau_samples

    def forward(self, x):
        h = self.body(x)
        taus = torch.rand(size=(h.shape[0], self.tau_samples), device=h.device)
        q = self.q(h, taus)
        return q, taus


class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=120):
        super().__init__()
        obs_dim, = obs_dim
        self.body = nn.Sequential(
            models.common.layer_init_truncated(nn.Linear(obs_dim, h_dim)),
            nn.ReLU())

        self.q = DuelingHead(h_dim, action_dim, h_dim=h_dim)

    def forward(self, x):
        h = self.body(x)
        q = self.q(h)
        return q


class DuelingHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, h_dim: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(input_dim),
            models.common.layer_init_truncated(nn.Linear(input_dim, h_dim)),
            nn.GELU())
        self.q = models.common.layer_init_truncated(nn.Linear(h_dim, output_dim))
        self.v = models.common.layer_init_truncated(nn.Linear(h_dim, 1))

    def forward(self, x):
        h = self.body(x)
        q = self.q(h)
        v = self.v(h)
        return v + q - q.mean(dim=-1, keepdim=True)


class AtariActorCritic(nn.Module):
    def __init__(self, observation_space: Tuple[int, ...], action_space: int = 6, h_dim: int = 256):
        super(AtariActorCritic, self).__init__()
        # TODO: verify init and layer norm
        self.body = models.common.AtariBody(observation_space)
        self.projection = nn.Sequential(nn.LayerNorm(self.body.output_dim),
                                        models.common.layer_init_truncated(nn.Linear(self.body.output_dim, h_dim)),
                                        nn.GELU())
        self.actor = models.common.layer_init_truncated(nn.Linear(h_dim, action_space))
        self.critic = models.common.layer_init_truncated(nn.Linear(h_dim, 1))

    def forward(self, x):
        x = x / 255.
        h = self.body(x)
        h = self.projection(h)
        return self.actor(h), self.critic(h)


class DuellingAtariNetwork(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...], action_dim: int, h_dim: int = 512):
        super().__init__()
        self.body = models.common.AtariBody(obs_dim)
        self.q = DuelingHead(self.body.output_dim, action_dim, h_dim=h_dim)

    def forward(self, s):
        h = self.body(s / 255.)
        return self.q(h)


class DistributionalAtariNetwork(nn.Module):
    def __init__(self, obs_dim: Tuple[int, ...], action_dim: int, h_dim: int = 512, n_tau_samples: int = 32):
        super().__init__()
        self.body = models.common.AtariBody(obs_dim)
        self.q = models.quantile_layers.ImplicitQuantileHead(self.body.output_dim, action_dim, d_model=h_dim)
        self.n_tau_samples = n_tau_samples

    def forward(self, s):
        h = self.body(s / 255.)
        taus = torch.rand(size=(h.shape[0], self.n_tau_samples), device=h.device)
        q = self.q(h, taus)
        return q, taus
