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
from rich.console import Console
from rich.progress import track
from rlmeta.agents.agent import Agent
from rlmeta.core.controller import Phase, ControllerLike
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import Action, TimeStep
from rlmeta.core.types import NestedTensor
from rlmeta.utils.stats_dict import StatsDict

import losses

console = Console()


class DistributedAgent(Agent):
    def __init__(self, controller: Optional[ControllerLike] = None, device: Optional[torch.device] = None):
        super(DistributedAgent, self).__init__()
        self._step_counter = 0
        self._controller = controller
        self._device = device

    def reset(self) -> None:
        self._step_counter = 0

    def eval(self,
             num_episodes: Optional[int] = None,
             keep_training_loops: bool = True) -> Optional[StatsDict]:
        if keep_training_loops:
            self._controller.set_phase(Phase.BOTH)
        else:
            self._controller.set_phase(Phase.EVAL)
        self._controller.reset_phase(Phase.EVAL, limit=num_episodes)
        while self._controller.count(Phase.EVAL) < num_episodes:
            time.sleep(1)
        stats = self._controller.stats(Phase.EVAL)
        return stats

    def device(self) -> torch.device:
        return self._device


class PPOAgent(DistributedAgent):

    def __init__(self,
                 model: ModelLike,
                 controller: Optional[ControllerLike] = None,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 256,
                 max_grad_norm: float = 0.5,
                 gamma: float = 0.99,
                 lambda_: float = 0.95,
                 entropy_coeff: float = 0.01,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 8,
                 rollout_length: int = 128,
                 deterministic_policy: bool = False,
                 ) -> None:

        super().__init__(controller)

        self._deterministic_policy = torch.tensor([deterministic_policy])
        self._model = model
        self._replay_buffer = replay_buffer

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._gamma = gamma
        self._lambda = lambda_
        self._entropy_coeff = entropy_coeff

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period
        self._local_batch_size = rollout_length

        self._trajectory = moolib.Batcher(self._local_batch_size, dim=0)
        self._last_transition = None
        self._device = None

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device

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

    def update(self) -> None:
        if self._replay_buffer is not None:
            if not self._trajectory.empty():
                replay = self._make_replay()
                self._send_replay(replay)

    async def async_update(self) -> None:
        if self._replay_buffer is not None:
            if not self._trajectory.empty():
                replay = await self._async_make_replay()
                await self._async_send_replay(replay)

    def _make_replay(self) -> List[NestedTensor]:
        s, a, r, d, pi_ref, values = self._trajectory.get()
        s, a, r, d, pi_ref = nested_utils.map_nested(lambda x: x.squeeze(-1)[:-1], (s, a, r, d, pi_ref))
        not_done = torch.logical_not(d)
        discount_t = (not_done.roll(1, dims=(0,)) * not_done) * self._gamma
        target_t = rlego.lambda_returns(r, discount_t, values.squeeze(-1)[1:], self._lambda)  # noqa
        return nested_utils.unbatch_nested(lambda x: x, (s, a, target_t, pi_ref), target_t.shape[0])

    async def _async_make_replay(self) -> List[NestedTensor]:
        return self._make_replay()

    def _send_replay(self, replay: List[NestedTensor]) -> None:
        self._replay_buffer.extend(replay)

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        await self._replay_buffer.async_extend(replay)

    def train(self, num_steps: int, writer) -> Optional[StatsDict]:
        self._controller.set_phase(Phase.TRAIN)
        self._replay_buffer.warm_up(self._learning_starts)
        stats = StatsDict()
        start_time = time.perf_counter()
        console.log(f"Training for num_steps = {num_steps}")
        for local_steps in track(range(num_steps), description="Training..."):
            t0 = time.perf_counter()
            _, batch, _ = self._replay_buffer.sample(self._batch_size)
            t1 = time.perf_counter()
            step_stats = self._train_step(batch)
            t2 = time.perf_counter()
            time_stats = {
                "sample_data_time": (t1 - t0) * 1000,
                "batch_learn_time": (t2 - t1) * 1000,
                "sps": local_steps / (time.perf_counter() - start_time),
            }
            stats.extend(step_stats)
            stats.extend(time_stats)

            writer.run.log({**step_stats, **time_stats})

            self._step_counter += 1
            if self._step_counter % self._model_push_period == 0:
                self._model.push()

        episode_stats = self._controller.stats(Phase.TRAIN)
        writer.run.log({k: v["mean"] for k, v in episode_stats.dict().items()})
        self._controller.reset_phase(Phase.TRAIN)

        return stats

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        s, a, v_target, pi_ref = nested_utils.map_nested(lambda x: x.to(self.device()), batch)
        loss, metrics = losses.ppo_loss(self._model, (s, a, v_target, pi_ref), entropy_cost=self._entropy_coeff)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        metrics['train/grad_norm'] = grad_norm

        self._optimizer.step()
        self._optimizer.zero_grad()

        return metrics


class ImpalaAgent(PPOAgent):

    def __init__(self,
                 model: ModelLike,
                 controller: Optional[ControllerLike] = None,
                 replay_buffer: Optional[ReplayBufferLike] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 batch_size: int = 48,
                 max_grad_norm: float = 0.5,
                 gamma: float = 0.99,
                 lambda_: float = 1.,
                 entropy_coeff: float = 0.01,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 10,
                 rollout_length: int = 20,
                 deterministic_policy: bool = False,
                 ) -> None:
        super().__init__(model, controller, replay_buffer, optimizer, batch_size, max_grad_norm, gamma, lambda_,
                         entropy_coeff, learning_starts, model_push_period, rollout_length, deterministic_policy)

        self._trajectory = moolib.Batcher(self._local_batch_size, dim=0)

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        obs = self._last_transition
        act, action_info = action
        next_obs, reward, done, _ = next_timestep
        self._trajectory.stack((obs, act, reward, next_obs, done, action_info["logpi"]))
        self._last_transition = next_obs.clone()

    def _make_replay(self) -> List[NestedTensor]:
        return [self._trajectory.get()]

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        batch = nested_utils.map_nested(lambda x: x.to(self.device()).squeeze(-1), batch)
        batch = losses.pre_process(batch, gamma=self._gamma, device=self.device())
        loss, metrics = losses.impala_loss(batch, model=self._model, lambda_=self._lambda)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        metrics['train/grad_norm'] = grad_norm

        self._optimizer.step()
        self._optimizer.zero_grad()

        return metrics
