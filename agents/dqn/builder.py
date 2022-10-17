import copy
from typing import Callable, Optional

import torch
from _rlmeta_extension import PrioritizedSampler
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike, ReplayBuffer
from rlmeta.storage import CircularBuffer

from agents.core import Builder
from agents.dqn.learning import ApexActor, ApexLearner, DistributionalApex
from models import AtariDQNModel, DistributionalAtariDQN


class ApexDQNAgentFactory(AgentFactory):

    def __init__(
            self,
            model: ModelLike,
            eps_func: Callable[[int], float],
            replay_buffer: Optional[ReplayBufferLike] = None,
            **kwargs,
    ) -> None:
        self._model = model
        self._eps_func = eps_func
        self._replay_buffer = replay_buffer
        self._kwargs = kwargs

    def __call__(self, index: int) -> ApexActor:
        model = self._make_arg(self._model, index)
        eps = self._eps_func(index)
        replay_buffer = self._make_arg(self._replay_buffer, index)
        return ApexActor(
            model,
            replay_buffer,
            eps,
            **self._kwargs
        )


class ConstantEpsFunc:

    def __init__(self, eps: float) -> None:
        self._eps = eps

    def __call__(self, index: int) -> float:
        return self._eps


class FlexibleEpsFunc:

    def __init__(self, eps: float, num: int, alpha: float = 7.0) -> None:
        self._eps = eps
        self._num = num
        self._alpha = alpha

    def __call__(self, index: int) -> float:
        if self._num == 1:
            return self._eps
        return self._eps ** (1.0 + index / (self._num - 1) * self._alpha)


class ApexDQNBuilder(Builder):
    def __init__(self, cfg):
        self.cfg = cfg
        self._learner_model = None
        self._actor_model = None

    def make_replay(self):
        rb = ReplayBuffer(
            CircularBuffer(self.cfg.agent.replay_buffer_size, collate_fn=torch.cat),
            PrioritizedSampler(priority_exponent=self.cfg.agent.priority_exponent))
        return rb

    def make_actor(self, model: ModelLike, rb: Optional[ReplayBufferLike] = None, deterministic: bool = False):
        if deterministic:
            eps_func = ConstantEpsFunc(self.cfg.agent.eval_eps)
        else:
            eps_func = FlexibleEpsFunc(self.cfg.agent.train_eps, self.cfg.training.num_rollouts)
        return ApexDQNAgentFactory(model, eps_func, rb, n_step=self.cfg.agent.n_step,
                                   rollout_length=self.cfg.agent.rollout_length)

    def make_learner(self, model: ModelLike, rb: ReplayBufferLike):
        optimizer = torch.optim.Adam(self._learner_model.parameters(), lr=self.cfg.optimizer.lr,
                                     eps=self.cfg.optimizer.eps)

        # optimizer = torch.optim.RMSprop(self._learner_model.parameters(), lr=0.00025 / 4, alpha=0.95, eps=1.5e-7)
        learner = DistributionalApex(
            model,
            replay_buffer=rb,
            optimizer=optimizer,
            learning_starts=self.cfg.agent.learning_starts,
        )
        return learner

    def make_network(self, env_spec):
        model = DistributionalAtariDQN(env_spec.observation_space.shape, env_spec.action_space.n).to(
            self.cfg.distributed.train_device)
        self._learner_model = model
        actor_model = copy.deepcopy(model).to(self.cfg.distributed.infer_device)
        for param in actor_model.parameters():
            param.requires_grad = False
        self._actor_model = actor_model
        return model
