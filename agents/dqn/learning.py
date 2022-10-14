import time
from typing import Optional, Tuple, List, Dict

import moolib
import rlego
import torch
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import TimeStep, Action, NestedTensor
from rlmeta.utils import nested_utils

from agents.core import Actor, Learner


class ApexActor(Actor):
    def __init__(self, model: ModelLike, replay_buffer: Optional[ReplayBufferLike] = None,
                 eps: float = 0.4,
                 n_step: int = 3,
                 rollout_length: int = 100,
                 gamma: float = 0.99):
        self._model = model
        self._replay_buffer = replay_buffer
        self._eps = torch.tensor((eps,), dtype=torch.float32)
        self._n_step = n_step
        self._trajectory = moolib.Batcher(rollout_length, dim=0, device="cpu")
        self._last_observation = None
        self._gamma = gamma

    async def async_act(self, timestep: TimeStep) -> Action:
        action, q, v = await self._model.async_act(timestep.observation, self._eps)
        return Action(action, info={"q": q, "v": v})

    async def async_observe_init(self, timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        self._last_observation = timestep.observation

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        obs = self._last_observation
        action, action_info = action
        next_obs, reward, done, _ = next_timestep
        self._trajectory.stack((obs, action, reward, done, action_info["q"], action_info["v"]))
        self._last_observation = next_obs

    async def async_update(self) -> None:
        if self._replay_buffer is None or self._trajectory.empty():
            return
        replay, priorities = self._make_replay()
        await self._replay_buffer.async_extend(replay, priorities)

    def _make_replay(self) -> Tuple[List[NestedTensor], torch.Tensor]:
        # TODO : this very likely is buggy make a test for it
        s_tm1, a_t, r_t, d_t, q_pi_t, q_star_t = self._trajectory.get()
        s_tm1, a_t, r_t, d_t = nested_utils.map_nested(lambda x: x.squeeze(-1)[:-1], (s_tm1, a_t, r_t, d_t))
        not_done = torch.logical_not(d_t)
        discount_t = (not_done.roll(1, dims=0) * torch.ones_like(not_done)) * self._gamma
        # replace the last element with q_star
        target = rlego.n_step_bootstrapped_returns(r_t, discount_t, q_star_t[1:].squeeze(dim=-1), n=self._n_step)
        # priorities = (target - q_pi_t[:-1].squeeze(dim=-1)).abs()
        priorities = (target - q_pi_t[:-1].squeeze(dim=-1)).abs()
        return nested_utils.unbatch_nested(lambda x: x, (s_tm1, a_t, target), target.shape[0]), priorities


class ApexLearner(Learner):

    def __init__(
            self,
            model: ModelLike,
            replay_buffer: ReplayBufferLike,
            optimizer: torch.optim.Optimizer,
            batch_size: int = 512,
            max_grad_norm: float = 40.0,
            importance_sampling_exponent: float = 0.4,
            target_sync_period: int = 2500,
            learning_starts: Optional[int] = None,
            model_push_period: int = 10,
    ) -> None:
        super().__init__()

        self._model = model

        self._replay_buffer = replay_buffer

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._importance_sampling_exponent = importance_sampling_exponent

        self._target_sync_period = target_sync_period

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period

        self._step_counter = 0
        self._update_priorities_future = None
        self._device = next(self._model.parameters()).device

    def train_step(self):
        # TODO: the first step takes 20x  longer than the rest. why?
        t0 = time.time()
        keys, batch, priorities = self._replay_buffer.sample(self._batch_size)
        t1 = time.time()
        metrics = self._train_step(keys, batch, priorities)
        t2 = time.time()

        self._step_counter += 1
        if self._step_counter % self._target_sync_period == 0:
            self._model.sync_target_net()

        if self._step_counter % self._model_push_period == 0:
            self._model.push()
        metrics["debug/replay_buffer_dt"] = (t1 - t0) * 1000
        metrics["debug/forward_dt"] = (t2 - t1) * 1000
        return metrics

    def prepare(self):
        if not self.can_train:
            self._replay_buffer.warm_up(self._learning_starts)
        self.can_train = True

    def _train_step(self, keys: torch.Tensor, batch: NestedTensor,
                    probabilities: torch.Tensor) -> Dict[str, float]:
        obs, action, target = nested_utils.map_nested(lambda x: x.to(self._device), batch)
        self._optimizer.zero_grad()

        q = self._model(obs)
        q = q.gather(1, action.unsqueeze(-1)).squeeze(-1)

        probabilities = probabilities.to(dtype=q.dtype, device=self._device)
        weight = probabilities.pow(-self._importance_sampling_exponent)
        weight.div_(weight.max())

        loss = 0.5 * ((target - q).pow(2) * weight).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        self._optimizer.step()

        with torch.no_grad():
            q = self._model(obs)
            q = q.gather(1, action.unsqueeze(-1)).squeeze(-1)
        priorities = (target - q).squeeze(-1).abs().cpu()

        # Wait for previous update request
        if self._update_priorities_future is not None:
            # print ("Waiting for previous update")
            self._update_priorities_future.wait()

        # Async update to start next training step when waiting for updating
        # priorities.
        self._update_priorities_future = self._replay_buffer.async_update(
            keys, priorities)

        return {
            "train_step/q": q.mean().item(),
            "train_step/target": target.mean().item(),
            "train_step/priorities": priorities.detach().mean().item(),
            "train_step/loss": loss.detach().mean().item(),
            "train_step/grad_norm": grad_norm.detach().mean().item(),
        }