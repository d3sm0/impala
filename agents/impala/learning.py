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

from agents.core import Actor, Learner


# try:
#     import functorch
#
#     batched_vtrace = functorch.vmap(rlego.vtrace_td_error_and_advantage)
# except ImportError:
def batched_vtrace(*batch):
    sequence = nested_utils.unbatch_nested(lambda x: x, batch, batch[0].shape[0])
    results = []
    for sub_seq in sequence:
        results.append(rlego.vtrace_td_error_and_advantage(*sub_seq))
    results = nested_utils.collate_nested(torch.stack, results)
    return results


#

class ImpalaActor(Actor):
    def __init__(self, model: ModelLike,
                 replay_buffer: ReplayBufferLike,
                 deterministic_policy: bool = False,
                 gamma: float = 0.99,
                 lambda_: float = 0.95,
                 rollout_length: int = 20
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
        self._trajectory.stack((obs, act, reward, next_obs, done, action_info["logpi"]))
        self._last_transition = next_obs.clone()

    async def async_update(self) -> None:
        if self._replay_buffer is not None:
            if not self._trajectory.empty():
                replay = await self._async_make_replay()
                await self._async_send_replay(replay)

    def _make_replay(self) -> List[NestedTensor]:
        s, a, r, _, d, pi_ref = self._trajectory.get()
        not_done = torch.logical_not(d)
        mask = torch.ones_like(not_done) * not_done.roll(1, dims=(0,))
        discount_t = (mask * not_done) * self._gamma
        return [s, a, r, discount_t, pi_ref]

    async def _async_make_replay(self) -> List[NestedTensor]:
        return self._make_replay()

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        await self._replay_buffer.async_append(replay)


class ImpalaLearner(Learner):

    def __init__(self,
                 model: ModelLike,
                 replay_buffer: ReplayBufferLike,
                 optimizer: torch.optim.Optimizer,
                 batch_size: int = 8,
                 max_grad_norm: float = 0.5,
                 entropy_coeff: float = 0.01,
                 learning_starts: Optional[int] = None,
                 model_push_period: int = 4) -> None:
        self._model = model
        self._replay_buffer = replay_buffer

        self._optimizer = optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm
        self._entropy_coeff = entropy_coeff

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period

        self._device = None
        self._step_count = 0

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device

    def prepare(self):
        self._replay_buffer.warm_up(self._learning_starts)

    def train_step(self):
        t0 = time.perf_counter()
        _, batch, _ = self._replay_buffer.sample(self._batch_size)
        n_samples = self._batch_size * batch[0][0].shape[0]
        batch = nested_utils.map_nested(lambda x: x.to(self.device()), batch)
        t1 = time.perf_counter()
        metrics = self._train_step(batch)
        t2 = time.perf_counter()
        update_time = 0
        self._step_count += 1
        if self._step_count % self._model_push_period == 0:
            start = time.perf_counter()
            self._model.push()
            update_time = time.perf_counter() - start
        metrics["debug/replay_sample_per_second"] = (n_samples / ((t1 - t0) * 1000))
        metrics["debug/gradient_per_second"] = (n_samples / ((t2 - t1) * 1000))
        metrics["debug/total_time"] = (time.perf_counter() - t0) * 1000
        metrics["debug/forward_dt"] = (t2 - t1) * 1000 / self._batch_size
        metrics["debug/update_time"] = update_time
        return metrics

    def _train_step(self, batch: NestedTensor) -> Dict[str, float]:
        self._optimizer.zero_grad(set_to_none=True)
        s, a, r, discount_t, pi_ref = nested_utils.collate_nested(lambda x: torch.stack(x).squeeze(dim=-1), batch)
        pi, values = self._model.forward(s.flatten(0, 1))
        pi = pi.reshape(s.shape[0], s.shape[1], -1)
        values = values.reshape(s.shape[0], s.shape[1])
        pi = torch.distributions.Categorical(logits=pi)
        pi_ref = torch.distributions.Categorical(logits=pi_ref)
        rho_tm1 = torch.exp(pi.log_prob(a) - pi_ref.log_prob(a))

        adv, err, _ = batched_vtrace(values[:, :-1], values[:, 1:],
                                     r[:, :-1],
                                     discount_t[:, :-1],
                                     rho_tm1[:, :-1])

        pg_loss = (pi.log_prob(a)[:, :-1] * adv).mean()
        value_loss = err.pow(2).mean()
        entropy_loss = pi.entropy().mean()

        loss = - pg_loss + value_loss - self._entropy_coeff * entropy_loss
        loss.backward()
        metrics = {

            "train/loss": loss.detach(),
            "train/entropy": entropy_loss,
            "train/td": value_loss.detach(),
            "train/pg": pg_loss.detach(),
            "train/kl": torch.distributions.kl_divergence(pi, pi_ref).mean().detach(),
            "train/ratio": rho_tm1.mean().detach(),

        }

        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(),
                                                   self._max_grad_norm)
        metrics['train/grad_norm'] = grad_norm

        self._optimizer.step()
        return metrics
