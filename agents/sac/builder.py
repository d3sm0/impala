import copy
from typing import Optional

import torch
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBuffer, ReplayBufferLike
from rlmeta.samplers import UniformSampler, PrioritizedSampler
from rlmeta.storage import CircularBuffer

from agents.core import Actor, Builder
from agents.sac.learning import SACActorRemote, SACLearner
from models.sac_model import SoftCritic, SoftActor


class SACFactory(AgentFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(SACActorRemote, *args, **kwargs)

    def __call__(self, index: int) -> Actor:
        return super().__call__(index)


class SACBuilder(Builder):
    def __init__(self, cfg):
        self.cfg = cfg
        self._learner_model = None
        self._actor_model = None

    def make_replay(self):
        sampler = UniformSampler()
        sampler.reset(self.cfg.training.seed)
        return ReplayBuffer(
            CircularBuffer(self.cfg.agent.replay_buffer_size, collate_fn=torch.cat),
            sampler
        )

    def make_actor(self, model: ModelLike, rb: Optional[ReplayBufferLike] = None, deterministic: bool = False):
        exploration_noise = 0. if deterministic else self.cfg.agent.exploration_noise
        return SACFactory(model, rb,  rollout_length=self.cfg.agent.rollout_length, exploration_noise=exploration_noise)

    def make_learner(self, model: ModelLike, rb: ReplayBufferLike):
        actor, critic = self._learner_model
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.agent.optimizer.critic_lr,
                                            eps=self.cfg.agent.optimizer.eps)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.agent.optimizer.actor_lr,
                                           eps=self.cfg.agent.optimizer.eps)
        return SACLearner(model, critic=critic, replay_buffer=rb,
                          batch_size=self.cfg.agent.batch_size,
                          critic_optimizer=critic_optimizer,
                          actor_optimizer=actor_optimizer,
                          model_push_period=self.cfg.agent.push_period,
                          learning_starts=self.cfg.agent.learning_starts,
                          )

    def make_network(self, env_spec):
        critic = SoftCritic(env_spec.observation_space.shape, env_spec.action_space.shape).to(
            self.cfg.distributed.train_device)
        actor = SoftActor(env_spec.observation_space.shape, env_spec.action_space.shape).to(
            self.cfg.distributed.train_device)
        self._learner_model = (actor, critic)
        inference_model = copy.deepcopy(actor).to(self.cfg.distributed.infer_device)
        for param in inference_model.parameters():
            param.requires_grad = False
        self._actor_model = inference_model
        return actor
