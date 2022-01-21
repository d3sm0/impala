# ported from https://github.com/deepmind/dm-haiku/blob/main/examples/impala_lite.py
import itertools
import queue
import threading
from typing import Any, NamedTuple, Callable, OrderedDict

import experiment_buddy.experiment_buddy
import gym
import numpy as np
import rlego
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.optim as optim
import tqdm
import tree
from torch._vmap_internals import vmap

import config
import models
import utils


class State(NamedTuple):
    reward: Any
    discount: Any
    observation: Any


class Transition(NamedTuple):
    state: State
    action: Any
    agent_out: Any


class Agent:
    """A simple, feed-forward agent."""

    def __init__(self, model: nn.Module, discount: float, device: torch.device):
        self._model = model
        self._discount = discount
        self._device = device

    def loss(self, trajs: Transition):
        """Computes a loss of trajs wrt params."""
        # Re-run the agent over the trajectories.
        obs = torch.Tensor(trajs.state.observation).to(self._device)
        T, B, *_ = obs.shape

        learner_policy, baseline_with_bootstrap = self._model(obs.flatten(0, 1))
        learner_logits = learner_policy.logits.view(T, B, -1)
        learner_logits = learner_logits[:-1]

        baseline_with_bootstrap = baseline_with_bootstrap.view(T, B)
        # Separate the bootstrap from the value estimates.
        baseline = baseline_with_bootstrap[:-1]
        baseline_tp1 = baseline_with_bootstrap[1:]

        # Remove bootstrap timestep from non-observations.
        _, actions, behavior_logits = tree.map_structure(lambda t: torch.tensor(t[:-1]), trajs)

        # Shift step_type/reward/discount back by one, so that actions match the
        # timesteps caused by the action.
        timestep = tree.map_structure(lambda t: t[1:], trajs.state)
        not_done = torch.Tensor(timestep.discount).to(self._device)
        discount = not_done * self._discount
        # The step is uninteresting if we transitioned LAST -> FIRST.
        mask = torch.roll(not_done, 1, dims=0)
        reward = torch.tensor(timestep.reward).to(self._device)

        learner_policy = torch_dist.Categorical(logits=learner_logits)
        learner_logits = learner_policy.log_prob(actions)
        behavior_logits = torch_dist.Categorical(logits=behavior_logits.squeeze(2)).log_prob(actions)
        rho = torch.exp(learner_logits - behavior_logits).detach()
        v_trace_fn = vmap(rlego.vtrace_td_error_and_advantage, in_dims=1, out_dims=1)
        adv, td, _ = v_trace_fn(baseline, baseline_tp1, reward, discount, rho)

        # Note that we use mean here, rather than sum as in canonical IMPALA.
        # Compute policy gradient loss.
        pg_loss = - (torch.tensor(mask) * rlego.vanilla_policy_gradient(learner_logits, adv)).sum(0).mean()

        # Baseline loss.
        td_loss = 0.5 * (torch.square(td) * torch.Tensor(mask)).sum(0).mean()
        ent_loss = learner_policy.entropy().sum(0).mean()

        actor_loss = pg_loss + 0.01 * ent_loss

        return (actor_loss, td_loss), {
            "pg_loss": float(pg_loss),
            "td_loss": float(td_loss),
            "ent_loss": float(ent_loss)
        }


class Learner:
    """Slim wrapper around an agent/optimizer pair."""

    def __init__(self, agent: Agent, actor_opt: torch.optim.Optimizer, critic_opt: torch.optim.Optimizer):
        self._agent = agent
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt

    def update(self, traj: Transition, clip: float = 5.):
        (actor_loss, critic_loss), extra = self._agent.loss(traj)
        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        self._critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self._agent._model.critic.parameters(), clip)  # noqa
        self._critic_opt.step()
        return extra


def run_actor(agent: models.Agent, get_params: Callable[[], OrderedDict[str, torch.Tensor]], enqueue_traj: Callable,
              enqueue_info: Callable[[dict], None], unroll_len: int, num_trajectories: int):
    """Runs an actor to produce num_trajectories trajectories."""
    env = utils.GymWrapper(gym.make(config.env_id))
    obs, *_ = env.reset()
    state = State(reward=0., discount=1., observation=obs)

    traj = []
    cumulative_reward = 0.
    for i in itertools.count():
        agent.load_state_dict(get_params())  # noqa
        # The first rollout is one step longer.
        for _ in range(unroll_len + int(i == 0)):
            with torch.no_grad():
                pi = agent.actor(state.observation.unsqueeze(0))
            action = pi.sample().squeeze()
            transition = Transition(state, action, pi.logits.squeeze(0))
            traj.append(transition)
            next_state, reward, done, _ = env.step(int(action.numpy()))
            cumulative_reward += reward
            state = State(reward=reward, observation=next_state, discount=(1 - done) * 1.)
            if done:
                enqueue_info({"cumulative_reward": cumulative_reward})
                cumulative_reward = 0
        # Stack and send the trajectory.
        stacked_traj = tree.map_structure(lambda *ts: np.stack(ts), *traj)
        enqueue_traj(stacked_traj)
        # Reset the trajectory, keeping the last timestep.
        traj = traj[-1:]


def run(writer: experiment_buddy.experiment_buddy.WandbWrapper):
    """Runs the example."""

    # Construct the agent network. We need a sample environment for its spec.
    env = gym.make(config.env_id)
    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    torch.manual_seed(config.seed)
    behavior = models.Agent(obs_dim, num_actions, config.h_dim)

    # Construct the agent and learner.
    net = models.Agent(obs_dim, num_actions, config.h_dim)
    agent = Agent(net, config.gamma, config.device)
    actor_opt = optim.Adam(net.actor.parameters(), config.critic_lr)
    critic_opt = optim.Adam(net.critic.parameters(), config.critic_lr)
    learner = Learner(agent, actor_opt, critic_opt)

    # Create accessor and queueing functions.
    current_params = lambda: net.state_dict()
    q = queue.Queue(maxsize=config.batch_size)
    q_info = queue.Queue()
    trajectories_per_actor = 500
    config.num_actors = 1

    def dequeue():
        batch = []
        for _ in range(config.batch_size):
            batch.append(q.get())
        batch = tree.map_structure(lambda *ts: np.stack(ts, axis=1), *batch)
        info = {}
        if not q_info.empty():
            info = q_info.get()
        return batch, info

    # Start the actors.
    for step in range(config.num_actors):
        args = (behavior, current_params, q.put, q_info.put, config.trajectory_len, trajectories_per_actor)
        threading.Thread(target=run_actor, args=args).start()

    # Run the learner.
    num_steps = config.max_steps
    for step in tqdm.trange(num_steps):
        traj, infos = dequeue()
        extra = learner.update(traj)
        writer.add_scalars({**extra, **infos}, global_step=step)


def main():
    experiment_buddy.register_defaults(config.__dict__)
    writer = experiment_buddy.deploy(host="", disabled=config.DEBUG)
    run(writer)


if __name__ == '__main__':
    main()
