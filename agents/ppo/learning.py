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
from agents.core import Actor, Learner


class PPOActor(Actor):
    def __init__(self, model: ModelLike,
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

    def act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action, logpi, v = self._model.act(obs, self._deterministic_policy)
        return Action(action, info={"logpi": logpi, "v": v})

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
        return nested_utils.unbatch_nested(lambda x: x, (s, a, target_t, pi_ref), target_t.shape[0])

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
        self._update_priorities_future = None

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device

    def prepare(self):
        self._replay_buffer.warm_up(self._learning_starts)

    def train_step(self):
        t0 = time.time()
        keys, batch, priorities = self._replay_buffer.sample(self._batch_size)
        t1 = time.time()
        metrics = self._train_step(batch, keys, priorities.float())
        t2 = time.time()
        self._step_counter += 1
        if self._step_counter % self._model_push_period == 0:
            self._model.push()

        metrics["debug/replay_buffer_dt"] = (t1 - t0) * 1000
        metrics["debug/forward_dt"] = (t2 - t1) * 1000
        return metrics

    def _train_step(self, batch: NestedTensor, keys, priorities) -> Dict[str, float]:
        (s, a, v_target, pi_ref), priorities = nested_utils.map_nested(lambda x: x.to(self.device()),
                                                                       (batch, priorities))
        pi_tm1, v_tm1 = self._model(s)
        weights = priorities.pow(-0.4)
        weights = weights / weights.max()
        td, adv, value_info = losses.value_loss(v_tm1, v_target, weights=weights)
        ppo_loss, policy_info = losses.ppo_loss(a, adv, pi_tm1, pi_ref, entropy_cost=self._entropy_coeff)
        loss = ppo_loss + td
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        metrics = {"loss": loss.item(), "grad_norm": grad_norm, **value_info, **policy_info}

        self._optimizer.step()
        self._optimizer.zero_grad()
        with torch.no_grad():
            pi_tm1, v_tm1 = self._model(s)
            adv = (v_target - v_tm1.squeeze(-1))
            log_prob = torch.distributions.Categorical(logits=pi_tm1).log_prob(a)
            priorities = (adv * log_prob).abs().cpu()
            priorities = priorities / (priorities.max() - priorities.min())
            metrics["priorities"] = priorities.mean().item()

        if self._update_priorities_future is not None:
            self._update_priorities_future.wait()
        self._update_priorities_future = self._replay_buffer.async_update(keys, priorities)

        return {f"train_step/{k}": v for k, v in metrics.items()}
