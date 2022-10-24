import copy
import time
from typing import List, Dict, Optional

import moolib
import torch
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import TimeStep, Action, NestedTensor
from rlmeta.utils import nested_utils

import agents.core


class SACActorRemote(agents.core.Actor):
    def __init__(self, model: ModelLike,
                 replay_buffer: ReplayBufferLike,
                 exploration_noise: float = 0.3,
                 gamma: float = 0.99,
                 rollout_length: int = 100,
                 ):
        self._model = model
        self._exploration_noise = torch.tensor([exploration_noise], dtype=torch.float32)
        self._gamma = gamma
        self._replay_buffer = replay_buffer
        self._rollout_length = rollout_length
        self._trajectory = moolib.Batcher(self._rollout_length, device='cpu', dim=0)
        self._last_transition = None

    async def async_act(self, timestep: TimeStep) -> Action:
        obs = timestep.observation
        action = await self._model.async_act(obs, self._exploration_noise)
        return Action(action)

    async def async_observe_init(self, timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        self._last_transition = timestep.observation

    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        if self._replay_buffer is None:
            return
        obs = self._last_transition
        act, action_info = action
        next_obs, reward, done, info = next_timestep
        # self._trajectory.append((obs, act, reward, next_obs, done))
        is_truncated = info['TimeLimit.truncated']
        mask = torch.argwhere(torch.tensor(is_truncated, dtype=torch.bool).to(done.device)).squeeze(1)
        done = done.scatter(0, mask, False)
        self._trajectory.stack((obs, act, reward, next_obs, done))
        self._last_transition = next_obs.clone()

    async def async_update(self) -> None:
        if self._replay_buffer is not None:
            if not self._trajectory.empty():
                replay = await self._async_make_replay()
                await self._async_send_replay(replay)

            # if self._last_transition.done:
            #    replay = await self._async_make_replay()
            #    await self._async_send_replay(replay)

            # if len(self._trajectory) > self._rollout_length or self._trajectory[-1][-2]:
            #     self._trajectory.clear()

    def _make_replay(self) -> List[NestedTensor]:
        # return self._trajectory
        # return self._trajectory
        s, a, r, s1, d = self._trajectory.get()
        # s, a, r, s1, d = self._trajectory
        # s, a, r, s1, d = nested_utils.collate_nested(torch.stack, self._trajectory)
        # TODO: save the last transition there  is a bug where the mask is not applied propery
        # m = (torch.logical_not(d) * torch.ones_like(d)).roll(1, dims=(0,))
        return list(nested_utils.unbatch_nested(lambda x: x.unsqueeze(1), (s, a, r, s1, d), self._rollout_length))

    async def _async_make_replay(self) -> List[NestedTensor]:
        return self._make_replay()

    async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
        await self._replay_buffer.async_extend(replay)

    # async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
    #    # await self._replay_buffer.async_extend(replay)
    #    n_sends = 5
    #    chunk_size, r = divmod(len(replay), n_sends)
    #    for i in range(n_sends):
    #        await self._replay_buffer.async_extend(replay[i * chunk_size:(i + 1) * chunk_size])
    #        await asyncio.sleep(0.1)
    #    if r > 0:
    #        await self._replay_buffer.async_extend(replay[-r:])

    # async def _async_send_replay(self, replay: List[NestedTensor]) -> None:
    #    batch = []
    #    while replay:
    #        batch.append(replay.pop())
    #        if len(batch) >= self._rollout_length:
    #            await self._replay_buffer.async_extend(batch)
    #            await asyncio.sleep(0.1 + random.random())
    #            batch.clear()
    #    if batch:
    #        await self._replay_buffer.async_extend(batch)
    #        batch.clear()


class SACLearner(agents.core.Learner):
    def __init__(self,
                 model: ModelLike,
                 critic: ModelLike,
                 target_actor: ModelLike,
                 replay_buffer: ReplayBufferLike,
                 critic_optimizer: torch.optim.Optimizer,
                 actor_optimizer: torch.optim.Optimizer,
                 batch_size: int = 256,
                 tune_alpha: bool = True,
                 max_grad_norm: Optional[float] = 40.,
                 epochs: int = 2,
                 policy_update_period: int = 2,
                 tau: float = 0.005,
                 learning_starts: Optional[int] = 1000,
                 model_push_period: int = 10,
                 ) -> None:
        super().__init__()

        self._model = model
        self._critic = critic
        self._replay_buffer = replay_buffer
        self._critic_optimizer = critic_optimizer
        self._actor_optimizer = actor_optimizer
        self._batch_size = batch_size
        self._max_grad_norm = max_grad_norm

        self._tune_alpha = tune_alpha
        self._tau = tau
        self._epochs = epochs
        self._policy_update_period = policy_update_period

        self._learning_starts = learning_starts
        self._model_push_period = model_push_period

        self._step_counter = 0
        self._device = None
        self._rb_cursor = None
        self._target_actor = copy.deepcopy(target_actor)
        # the critic is local citizen hence share memory
        # self._critic.share_memory()

    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._device

    def prepare(self):
        if self.can_train is False:
            self._replay_buffer.warm_up(self._learning_starts)
        self.can_train = True

    def train_step(self):
        t0 = time.perf_counter()
        _, batch, _ = self._replay_buffer.sample(self._batch_size)
        batch = nested_utils.map_nested(lambda x: x.to(self.device()).squeeze(dim=-1), batch)
        t1 = time.perf_counter()
        metrics = self._train_critic(batch)
        actor_metrics = self._train_actor(batch)
        metrics.update(actor_metrics)
        if self._tune_alpha:
            alpha_metrics = self._train_alpha(batch)
            metrics.update(alpha_metrics)
        # if self._step_counter % self._policy_update_period == 0:
        #     for _ in range(self._policy_update_period):
        #         metrics.update(actor_metrics)
        #         # alpha_metrics = self._train_alpha(batch)
        t2 = time.perf_counter()
        update_time = 0
        if self._step_counter % self._model_push_period == 0:
            start = time.perf_counter()
            self._model.push()
            update_time = time.perf_counter() - start

        capacity, size = self._replay_buffer.info()
        metrics["debug/rb_capacity"] = capacity
        # if self._step_counter % 100 == 0:
        #     with torch.no_grad():
        #         self._critic.target_critic.load_state_dict(self._critic.critic.state_dict())
        #         # self._target_actor.load_state_dict(self._model.state_dict())
        with torch.no_grad():
            for param, target_param in zip(self._critic.critic.parameters(), self._critic.target_critic.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

            for param, target_param in zip(self._model.parameters(), self._target_actor.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        self._step_counter += 1

        metrics["debug/replay_sample_per_second"] = (self._batch_size / ((t1 - t0) * 1000))
        metrics["debug/gradient_per_second"] = (self._batch_size / ((t2 - t1) * 1000))
        metrics["debug/total_time"] = (time.perf_counter() - t0) * 1000
        metrics["debug/sample_dt"] = (t1 - t0) * 1000
        metrics["debug/forward_dt"] = (t2 - t1) * 1000
        metrics["debug/update_dt"] = update_time * 1000
        return metrics

    def _train_critic(self, batch: NestedTensor) -> Dict[str, float]:
        loss, metrics = critic_loss(self._target_actor, self._critic, batch)
        # Do not move this after the loss because actor loss generates a gradient for the critic
        self._critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._max_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
            metrics["train/critic_grad_norm"] = total_norm
        self._critic_optimizer.step()
        return metrics

    def _train_actor(self, batch: NestedTensor) -> Dict[str, float]:
        # hook = self._critic.register_full_backward_hook(clip_dqda)
        loss, metrics = actor_loss(self._model._wrapped, self._critic, batch)
        self._actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # hook.remove()
        if self._max_grad_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            metrics["train/actor_grad_norm"] = total_norm
        self._actor_optimizer.step()
        return metrics

    def _train_alpha(self, batch: NestedTensor) -> Dict[str, float]:
        loss, metrics = alpha_loss(self._model, self._critic, batch)
        self._critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._critic_optimizer.step()
        return metrics


def clip_dqda(m, grad_input, grad_output):
    dq_ds, dq_da = grad_input
    dq_da = dq_da / dq_da.norm(p=2, dim=1, keepdim=True).clamp_max(1)
    return (dq_ds, dq_da)


def critic_loss(actor, critic, batch, gamma=0.99):
    s, a, r, s1, d = batch

    with torch.no_grad():
        next_state_actions, next_state_log_pi, _ = actor.policy(s1)
        qf1_next_target, qf2_next_target = critic.target_critic(s1, next_state_actions)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - critic.alpha * next_state_log_pi
        next_q_value = r + d.logical_not() * gamma * min_qf_next_target
        # next_q_value = rlego.discounted_returns(r, torch.logical_not(d) * gamma, min_qf_next_target)
    qf1, qf2 = critic(s, a)

    qf1_loss = (next_q_value - qf1).pow(2).mean()
    qf2_loss = (next_q_value - qf2).pow(2).mean()
    qf_loss = qf1_loss + qf2_loss
    return qf_loss, {"train/qf1_loss": qf1_loss, "train/qf2_loss": qf2_loss, "train/qf1": qf1.mean(),
                     "train/qf2": qf2.mean(), "train/qf_loss": qf_loss / 2.}


def actor_loss(actor, critic, batch):
    s, a, r, s1, d = batch
    pi, log_pi, std = actor.policy(s)
    qf1_pi, qf2_pi = critic(s, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((critic.alpha * log_pi) - min_qf_pi).mean()
    return actor_loss, {"train/actor_loss": actor_loss, "train/actor_std": std.mean(dim=-1).mean()}


def alpha_loss(actor, critic, batch):
    s, a, r, s1, d = batch
    with torch.no_grad():
        _, log_pi, _ = actor.policy(s)
    alpha_loss = - (critic.log_alpha * (log_pi + critic.target_entropy)).mean()
    return alpha_loss, {"train/alpha_loss": alpha_loss, "train/alpha": critic.alpha}
