from typing import Tuple

import rlego
import torch
import torch.nn as nn


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self._add_action = nn.Sequential(nn.Linear(h_dim + action_dim, h_dim), nn.ReLU())
        self._out = nn.Linear(h_dim, 1)

    def forward(self, state, action):
        h = self._critic(state)
        state_and_action = torch.cat([h, action], dim=-1)
        h = self._add_action(state_and_action)
        return self._out(h).squeeze(-1)


class VFunction(nn.Module):
    def __init__(self, obs_dim, h_dim=100):
        super().__init__()

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, state):
        return self._critic(state).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, h_dim=100):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            rlego.SoftmaxPolicy(h_dim, action_dim))

    def forward(self, s):
        out = self.actor(s)
        return out


class Agent(nn.Module):
    """A simple network."""

    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 32):
        super().__init__()
        self._num_actions = action_dim
        self.actor = Actor(obs_dim, action_dim, h_dim)
        self.critic = VFunction(obs_dim, h_dim)

    def forward(self, observation) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of observations."""
        hidden = observation
        policy = self.actor(hidden)
        baseline = self.critic(hidden)
        return policy, baseline
