from typing import Tuple

import rlego
import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_gain: int = 1):
    def init_func(m):
        if hasattr(m, "weight"):
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()

        self._critic = nn.Sequential(
            # nn.Identity(),
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
        )
        self._add_action = nn.Sequential(nn.Linear(h_dim + action_dim, 1))
        # self._out = nn.Linear(h_dim, 1)

    def forward(self, state, action):
        h = self._critic(state)
        state_and_action = torch.cat([h, action], dim=-1)
        q = self._add_action(state_and_action)
        return q.squeeze(dim=-1)


class VFunction(nn.Module):
    def __init__(self, obs_dim, h_dim=100):
        super().__init__()

        self._critic = nn.Sequential(
            # nn.Identity(),
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, state):
        return self._critic(state).squeeze(-1)


class Q_and_V(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU()
        )
        self.q = QFunction(obs_dim, action_dim, h_dim)
        self.critic = VFunction(obs_dim, h_dim)

    def forward(self, state, action):
        h = self.body(state)
        q = self.q(h, action)
        v = self.critic(h)
        return q, v


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            rlego.GaussianPolicy(h_dim, action_dim)
        )

    def forward(self, s):
        out = self.actor(s)
        return out


class Body(nn.Module):
    def __init__(self, obs_dim, h_dim=100):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.SiLU())

    def forward(self, s):
        out = self.body(s)
        return out


class Agent(nn.Module):
    """A simple network."""

    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 32):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, h_dim)
        # self.q_and_v = Q_and_V(obs_dim, action_dim, h_dim)
        self.q = QFunction(obs_dim, action_dim, h_dim)
        self.critic = VFunction(obs_dim, h_dim)
        init_weights(self.critic)
        init_weights(self.q)
        init_weights(self.actor)

    def forward(self, observation) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = observation
        policy = self.actor(hidden)
        baseline = self.critic(hidden)
        return policy, baseline
