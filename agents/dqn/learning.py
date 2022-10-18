import time
from typing import Optional, Tuple, List, Dict

# import functorch
import moolib
import rlego
import torch
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import TimeStep, Action, NestedTensor
from rlmeta.utils import nested_utils

from agents.core import Actor, Learner
from models.quantile_layers import quantile_regression_loss


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
        priorities = (target - q_pi_t[:-1].squeeze(dim=-1)).abs()
        return nested_utils.unbatch_nested(lambda x: x.unsqueeze(1), (s_tm1, a_t, target), target.shape[0]), priorities


class ApexDistributionalActor(ApexActor):
    def _make_replay(self) -> Tuple[List[NestedTensor], torch.Tensor]:
        # TODO: we should use vmap here but is not supported by the cluster
        # if so ApexActor and ApexDistributional should be merged
        s_tm1, a_t, r_t, d_t, q_pi_t, q_star_t = self._trajectory.get()
        s_tm1, a_t, r_t, d_t = nested_utils.map_nested(lambda x: x.squeeze(-1)[:-1], (s_tm1, a_t, r_t, d_t))
        not_done = torch.logical_not(d_t)
        discount_t = (not_done.roll(1, dims=0) * torch.ones_like(not_done)) * self._gamma
        # replace the last element with q_star
        n, k = q_star_t.shape
        targets = []
        for idx in range(k):
            target = rlego.n_step_bootstrapped_returns(r_t, discount_t, q_star_t[1:, idx], n=self._n_step)
            targets.append(target)
        target = torch.stack(targets, dim=1)
        priorities = (target.mean(1) - q_pi_t[:-1].squeeze(dim=-1)).abs()
        return nested_utils.unbatch_nested(lambda x: x.unsqueeze(1), (s_tm1, a_t, target), target.shape[0]), priorities


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
        t0 = time.perf_counter()
        keys, batch, priorities = self._replay_buffer.sample(self._batch_size)
        batch = nested_utils.map_nested(lambda x: x.to(self._device), batch)
        t1 = time.perf_counter()
        metrics = self._train_step(keys, batch, priorities)
        t2 = time.perf_counter()
        self._step_counter += 1

        target_time = 0
        if self._step_counter % self._target_sync_period == 0:
            target_time = time.perf_counter()
            self._model.sync_target_net()
            target_time = time.perf_counter() - target_time
        update_time = 0

        if self._step_counter % self._model_push_period == 0:
            start_time = time.perf_counter()
            self._model.push()
            update_time = time.perf_counter() - start_time

        metrics["debug/replay_sample_per_second"] = self._batch_size / ((t1 - t0) * 1000)
        metrics["debug/forward_dt"] = (t2 - t1) * 1000
        metrics["debug/update_time"] = update_time
        metrics["debug/target_sync_time"] = target_time
        metrics["debug/gradient_per_second"] = (self._batch_size / ((t2 - t0) * 1000))
        metrics["debug/total_time"] = (time.perf_counter() - t0) * 1000
        return metrics

    def prepare(self):
        if not self.can_train:
            self._replay_buffer.warm_up(self._learning_starts)
        self.can_train = True

    def _train_step(self, keys: torch.Tensor, batch: NestedTensor,
                    probabilities: torch.Tensor) -> Dict[str, float]:
        self._optimizer.zero_grad()
        obs, action, target = batch
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
            priorities = (target - q).squeeze(-1).abs().cpu()

        # Wait for previous update request
        priority_update_time = 0
        if self._update_priorities_future is not None:
            t0 = time.perf_counter()
            self._update_priorities_future.wait()
            priority_update_time = (time.perf_counter() - t0) * 1000

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
            "debug/priority_update_time": priority_update_time,
        }


class DistributionalApex(ApexLearner):

    def _train_step(self, keys: torch.Tensor, batch: NestedTensor,
                    probabilities: torch.Tensor) -> Dict[str, float]:
        self._optimizer.zero_grad()
        obs, action, target = batch
        q, taus = self._model(obs)
        q = q.gather(-1, action.reshape(-1, 1, 1).repeat(1, taus.shape[-1], 1)).squeeze(-1)

        probabilities = probabilities.to(dtype=q.dtype, device=self._device)
        weight = probabilities.pow(-self._importance_sampling_exponent)
        weight.div_(weight.max())

        loss = quantile_regression_loss(q, taus, target).mul(weight).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        self._optimizer.step()
        with torch.no_grad():
            priorities = (target.mean(dim=-1) - q.mean(dim=-1)).abs().cpu()

        # Wait for previous update request
        priority_update_time = 0
        if self._update_priorities_future is not None:
            t0 = time.perf_counter()
            self._update_priorities_future.wait()
            priority_update_time = (time.perf_counter() - t0) * 1000

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
            "debug/priority_update_time": priority_update_time,
        }
