# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Dict, List, Optional

import moolib
import rlego
import rlmeta.utils.nested_utils as nested_utils
import torch
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor

import losses
import models.models
from agents.core import Actor, Learner


class PPOActorRemote(Actor):
    def __init__(self, model: models.models.AtariActorCritic,
                 replay_buffer: ReplayBufferLike,
                 deterministic_policy: bool = False,
                 gamma: float = 0.99,
                 lambda_: float = 0.95,
                 rollout_length: int = 128
                 ):
        self._gamma = gamma
        self._lambda = lambda_
        self._deterministic_policy = torch.tensor([deterministic_policy])
        self._model = model
        self._replay_buffer = replay_buffer
        self._trajectory = moolib.Batcher(rollout_length, dim=0, device="cpu")
        self._last_transition = None

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = await self._model.async_act(
            obs, self._deterministic_policy)
        return Action(action, info={"logpi": logpi, "v": v})

    async def async_observe_init(self, timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        obs, _, _, _ = timestep
        self._last_transition = obs.clone()

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        obs = self._last_transition
        act, action_info = action
        next_obs, reward, done, _ = next_timestep
        self._trajectory.stack((obs, act, reward, done, action_info["logpi"], action_info["v"]))
        self._last_transition = next_obs.clone()

    async def async_update(self) -> None:
        if self._replay_buffer is not None:
            if not self._trajectory.empty():
                replay = await self._async_make_replay()
                await self._async_send_replay(replay)

    def _make_replay(self) -> List[NestedTensor]:
        s, a, r, d, pi_ref, values = self._trajectory.get()
        s, a, r, d, pi_ref = nested_utils.map_nested(lambda x: x.squeeze(-1)[:-1], (s, a, r, d, pi_ref))
        not_done = torch.logical_not(d)
        mask = torch.ones_like(not_done) * not_done.roll(1, dims=(0,))
        discount_t = (mask * not_done) * self._gamma
        target_t = rlego.lambda_returns(r, discount_t, values.squeeze(-1)[1:], self._lambda)  # noqa
        return nested_utils.unbatch_nested(lambda x: x.unsqueeze(1), (s, a, target_t, pi_ref), target_t.shape[0])

    async def _async_make_replay(self) -> List[NestedTensor]:
        return self._make_replay()

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        await self._replay_buffer.async_extend(replay)


class PPOLearner(Learner):

    def __init__(self,
                 model: ModelLike,
                 replay_buffer: ReplayBufferLike,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int = 256,
                 max_grad_norm: float = 0.5,
                 entropy_coeff: float = 0.01,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 8) -> None:
        self._model = model
        self._replay_buffer = replay_buffer

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm
        self._entropy_coeff = entropy_coeff

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period

        self._device = None
        self._step_counter = 0

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device

    def prepare(self):
        if not self.can_train:
            self._replay_buffer.warm_up(self._learning_starts)
        self.can_train = True

    def train_step(self):
        t0 = time.perf_counter()
        _, batch, _ = self._replay_buffer.sample(self._batch_size)
        batch = nested_utils.map_nested(lambda x: x.to(self.device()), batch)
        t1 = time.perf_counter()
        metrics = self._train_step(batch)
        t2 = time.perf_counter()
        self._step_counter += 1
        update_time = 0
        if self._step_counter % self._model_push_period == 0:
            start = time.perf_counter()
            self._model.push()
            update_time = (time.perf_counter() - start) * 1000
        metrics["debug/replay_sample_per_second"] = (self._batch_size / ((t1 - t0) * 1000))
        metrics["debug/gradient_per_second"] = (self._batch_size / ((t2 - t1) * 1000))
        metrics["debug/total_time"] = (time.perf_counter() - t0) * 1000
        metrics["debug/forward_dt"] = (t2 - t1) * 1000
        metrics["debug/update_time"] = update_time
        return metrics

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        s, a, v_target, pi_ref = batch
        loss, metrics = losses.ppo_loss(self._model, (s, a, v_target, pi_ref), entropy_cost=self._entropy_coeff)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        metrics['train_step/grad_norm'] = grad_norm

        self._optimizer.step()
        self._optimizer.zero_grad()

        return metrics
