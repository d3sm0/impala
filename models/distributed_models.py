import copy
from typing import Tuple

import torch
from rlmeta.core import remote
from rlmeta.core.model import RemotableModel

import models


class AtariPPOModel(RemotableModel):

    def __init__(self, observation_space: Tuple[int, ...], action_dim: int) -> None:
        super().__init__()
        self.model = models.models.AtariActorCritic(observation_space, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p, v = self.model(obs)
        return p, v

    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor, deterministic_policy: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device

        with torch.inference_mode():
            x = obs.to(device)
            logits, v = self.forward(x)
            action = logits.softmax(dim=-1).multinomial(1, replacement=True)
            action = torch.where(deterministic_policy.to(device), logits.argmax(dim=-1, keepdim=True), action)

        return action.cpu(), logits.cpu(), v.cpu()


class DistributionalAtariDQN(RemotableModel):
    def __init__(self, observation_space: Tuple[int, ...], action_dim: int, n_tau_samples: int = 32) -> None:
        super().__init__()
        self.online_net = models.models.DistributionalAtariNetwork(observation_space, action_dim,
                                                                   n_tau_samples=n_tau_samples)
        self.target_net = copy.deepcopy(self.online_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_net(obs)

    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor,
            eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = obs.to(device)
            eps = eps.to(device)  # size = (batch_size, 1)
            q, _ = self.online_net(x)  # size = (batch_size, action_dim)
            q = q.mean(1)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)
            pi = torch.ones_like(q) * (eps / (action_dim - 1))
            pi.scatter_(dim=-1, index=greedy_action, src=1.0 - eps)
            action = pi.multinomial(1, replacement=True)
            q_pi = q.gather(dim=-1, index=action)
            q_target, taus = self.target_net(x)
            q_star = q_target.gather(dim=-1, index=greedy_action.unsqueeze(1).repeat(1, q_target.shape[1], 1)).squeeze(
                -1)

        return action.cpu(), q_pi.cpu(), q_star.cpu()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())


class AtariDQNModel(RemotableModel):

    def __init__(self, observation_space: Tuple[int, ...], action_dim: int) -> None:
        super().__init__()
        self.online_net = models.models.DuellingAtariNetwork(observation_space, action_dim)
        self.target_net = copy.deepcopy(self.online_net)
        for p in self.target_net.parameters():
            p.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_net(obs)

    @remote.remote_method(batch_size=128)
    def act(self, obs: torch.Tensor,
            eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        with torch.no_grad():
            x = obs.to(device)
            eps = eps.to(device)  # size = (batch_size, 1)
            q = self.online_net(x)  # size = (batch_size, action_dim)
            _, action_dim = q.size()
            greedy_action = q.argmax(-1, keepdim=True)
            pi = torch.ones_like(q) * (eps / (action_dim - 1))
            pi.scatter_(dim=-1, index=greedy_action, src=1.0 - eps)
            action = pi.multinomial(1, replacement=True)
            q_pi = q.gather(dim=-1, index=action)
            q_target = self.target_net(x)
            q_star = q_target.gather(dim=-1, index=greedy_action)

        return action.cpu(), q_pi.cpu(), q_star.cpu()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
