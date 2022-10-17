import copy
from typing import Optional

import torch
from rlmeta.samplers import UniformSampler
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBuffer, ReplayBufferLike
from rlmeta.storage import CircularBuffer

from agents.core import Actor, Builder
from agents.ppo.learning import PPOActor, PPOLearner
from models import AtariPPOModel


class PPOFactory(AgentFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(PPOActor, *args, **kwargs)

    def __call__(self, index: int) -> Actor:
        return super().__call__(index)


class PPOBuilder(Builder):
    def __init__(self, cfg):
        self.cfg = cfg
        self._learner_model = None
        self._actor_model = None

    def make_replay(self):
        return ReplayBuffer(
            CircularBuffer(self.cfg.agent.replay_buffer_size, collate_fn=torch.cat),
            UniformSampler()
        )

    def make_actor(self, model: ModelLike, rb: Optional[ReplayBufferLike] = None, deterministic: bool = False):
        return PPOFactory(model, rb, deterministic)

    def make_learner(self, model: ModelLike, rb: ReplayBufferLike):
        optimizer = torch.optim.Adam(self._learner_model.parameters(), lr=self.cfg.optimizer.lr,
                                     eps=self.cfg.optimizer.eps)
        return PPOLearner(model, rb, optimizer)

    def make_network(self, env_spec):
        model = AtariPPOModel(env_spec.observation_space, env_spec.action_space.n).to(self.cfg.distributed.train_device)
        self._learner_model = model
        actor_model = copy.deepcopy(model).to(self.cfg.distributed.infer_device)
        for param in actor_model.parameters():
            param.requires_grad = False
        self._actor_model = actor_model
        return model
