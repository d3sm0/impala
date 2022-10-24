from typing import Optional

import rlmeta.core.replay_buffer
import torch
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.samplers import UniformSampler
from rlmeta.storage import CircularBuffer

import agents.core
from agents.sac.learning import SACActorRemote, SACLearner
from models.sac_model import SoftCritic, SoftActor


class SACFactory(AgentFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(SACActorRemote, *args, **kwargs)

    def __call__(self, index: int) -> agents.core.Actor:
        return super().__call__(index)


class SACBuilder(agents.core.Builder):
    def __init__(self, cfg):
        self.cfg = cfg
        self._learner_model = None
        self._actor_model = None

    def make_replay(self):
        sampler = UniformSampler()
        sampler.reset(self.cfg.training.seed)
        return rlmeta.core.replay_buffer.ReplayBuffer(
            CircularBuffer(self.cfg.agent.replay_buffer_size, collate_fn=torch.cat),
            sampler
        )

    def make_actor(self, model: ModelLike, rb: Optional[ReplayBufferLike] = None, deterministic: bool = False):
        exploration_noise = 0. if deterministic else self.cfg.agent.exploration_noise
        return SACFactory(model, rb, rollout_length=self.cfg.agent.rollout_length, exploration_noise=exploration_noise)

    def make_learner(self, model: ModelLike, rb: ReplayBufferLike):
        actor, critic = self._learner_model
        critic_optimizer = torch.optim.Adam([{"params": critic.critic.parameters()}, {"params": critic.log_alpha}],
                                            lr=self.cfg.agent.optimizer.critic_lr, eps=self.cfg.agent.optimizer.eps)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.agent.optimizer.actor_lr,
                                           eps=self.cfg.agent.optimizer.eps)

        return SACLearner(model, critic=critic,
                          target_actor=actor,
                          replay_buffer=rb,
                          batch_size=self.cfg.agent.batch_size,
                          critic_optimizer=critic_optimizer,
                          actor_optimizer=actor_optimizer,
                          model_push_period=self.cfg.agent.push_period,
                          learning_starts=self.cfg.agent.learning_starts,
                          tune_alpha=self.cfg.agent.tune_alpha,
                          )

    def make_network(self, env_spec):
        critic = SoftCritic(env_spec.observation_space.shape, env_spec.action_space.shape,
                            alpha=self.cfg.agent.alpha).to(
            self.cfg.distributed.train_device)
        actor = SoftActor(env_spec.observation_space.shape, env_spec.action_space.shape).to(
            self.cfg.distributed.train_device)
        self._learner_model = (actor, critic)
        inference_model = SoftActor(env_spec.observation_space.shape, env_spec.action_space.shape).to(
            self.cfg.distributed.infer_device)
        inference_model.load_state_dict(actor.state_dict())
        for p in inference_model.parameters():
            p.requires_grad = False
        self._actor_model = inference_model

        return actor
