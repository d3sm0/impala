import functools
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from rlmeta.core import remote
from rlmeta.core.model import RemotableModel

import models.common

LOG_STD_MAX = 2
LOG_STD_MIN = -5


# TODO: verfiy init and head

def to_action(mean, log_std, action_scale=1.0, action_bias=0.):
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()
    y_t = torch.tanh(x_t)
    action = y_t * action_scale + action_bias
    log_prob = normal.log_prob(x_t)
    # Enforcing Action Bound
    log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
    log_prob = log_prob.sum(-1)
    return action, log_prob, std


init_ = models.common.layer_init_uniform
act = nn.Tanh

near_zero_init = functools.partial(models.common.layer_init_truncated, scale=1e-4)


class D4PGCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(D4PGCritic, self).__init__()
        self.body = nn.Sequential(
            init_(nn.Linear(np.prod(observation_space) + np.prod(action_space), 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            init_(nn.Linear(256, 256)),
            nn.Tanh(),
            init_(nn.Linear(256, 1))
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.body(x).squeeze(dim=-1)


class D4PGACtorBody(nn.Module):
    def __init__(self, observation_space: Tuple[int, ...]):
        super(D4PGACtorBody, self).__init__()
        self.body = nn.Sequential(
            init_(nn.Linear(np.prod(observation_space), 256)),
            nn.LayerNorm(256),
            nn.Tanh(),
            init_(nn.Linear(256, 256)),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.body(x)
        return h

    @property
    def output_dim(self):
        return 256


class SoftQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.body = nn.Sequential(
            init_(nn.Linear(np.array(observation_space).prod() + np.prod(action_space), 256)),
            nn.GELU(),
            init_(nn.Linear(256, 256)),
            nn.GELU(),
            init_(nn.Linear(256, 1))
        )

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = self.body(x)
        return x.squeeze(-1)


class ActorBody(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        self.body = nn.Sequential(
            init_(nn.Linear(np.array(observation_space).prod(), 256)),
            nn.GELU(),
            init_(nn.Linear(256, 256)),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.body(x)
        return x

    @property
    def output_dim(self):
        return 256


class Actor(nn.Module):
    def __init__(self, observation_space, action_space: Tuple[int, ...], action_scale=1.0, action_bias=0.):
        super(Actor, self).__init__()

        self.body = ActorBody(observation_space)
        self.head = ContionusHead(self.body.output_dim, action_space, action_scale, action_bias)
        self.action_scale = self.head.action_scale
        self.action_bias = self.head.action_bias

    def forward(self, x):
        return self.head(self.body(x))


class ContionusHead(nn.Module):
    def __init__(self, in_features: int, action_space: Tuple[int, ...], action_scale=1.0, action_bias=0.):
        super(ContionusHead, self).__init__()
        self.fc_mean = init_(nn.Linear(in_features, np.prod(action_space)))
        self.fc_logstd = init_(nn.Linear(in_features, np.prod(action_space)))
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, h):
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)
        # Try elu instead of this
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.q1 = SoftQNetwork(observation_space, action_space)
        self.q2 = SoftQNetwork(observation_space, action_space)
        # self.q1 = D4PGCritic(observation_space, action_space)
        # self.q2 = D4PGCritic(observation_space, action_space)

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)


class SoftCritic(nn.Module):
    def __init__(self, observation_space, action_space, alpha=1.):
        super().__init__()
        self.critic = Critic(observation_space, action_space)
        self.target_critic = Critic(observation_space, action_space)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.register_parameter("log_alpha",
                                nn.Parameter(torch.tensor(math.log(alpha), dtype=torch.float32), requires_grad=True))
        self.register_buffer("target_entropy", nn.Parameter(torch.tensor(-np.prod(action_space), dtype=torch.float32),
                                                            requires_grad=False))
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # self.share_memory()

    def forward(self, s, a):
        return self.critic.q1(s, a), self.critic.q2(s, a)

    def target(self, s, a):
        with torch.no_grad():
            return self.target_critic.q1(s, a), self.target_critic.q2(s, a)

    @property
    def alpha(self):
        return self.log_alpha.exp()


class SoftActor(RemotableModel):
    def __init__(self, observation_space, action_space, action_scale=1., action_bias=0.):
        super(SoftActor, self).__init__()
        self.actor = Actor(observation_space, action_space, action_scale, action_bias)

    def forward(self, x):
        return self.actor.forward(x)

    # This should probably match number of environments someohow
    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor, eps: torch.Tensor = 0.) -> torch.Tensor:
        device = next(self.parameters()).device
        eps = eps.to(device)
        with torch.no_grad():
            mu, log_std = self.forward(obs.to(device))
            std = log_std.exp()
            x_t = mu + std * torch.randn_like(std) * eps
            action = torch.tanh(x_t) * self.actor.action_scale + self.actor.action_bias
        return action.cpu()

    def policy(self, s):
        mu, log_std = self.actor(s)
        return to_action(mu, log_std)
