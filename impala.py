# ported from https://github.com/deepmind/dm-haiku/blob/main/examples/impala_lite.py
import queue
import threading
from typing import Any, NamedTuple, Tuple, Callable, OrderedDict

import experiment_buddy
import gym
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.optim as optim
import tree
from rlego.src import vtrace
from torch._vmap_internals import vmap
from tqdm import trange


class State(NamedTuple):
    reward: Any
    discount: Any
    observation: Any


class Transition(NamedTuple):
    state: State
    action: Any
    agent_out: Any


class SimpleNet(nn.Module):
    """A simple network."""

    def __init__(self, obs_dim: int, num_actions: int, h_dim: int = 32):
        super().__init__()
        self._num_actions = num_actions
        self.policy = nn.Sequential(nn.Linear(obs_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh(),
                                    nn.Linear(h_dim, self._num_actions))
        self.baseline = nn.Sequential(nn.Linear(obs_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh(),
                                      nn.Linear(h_dim, 1))

    def forward(self, observation) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of observations."""
        hidden = observation
        policy_logits = self.policy(hidden)
        baseline = self.baseline(hidden)
        baseline = torch.squeeze(baseline, dim=-1)
        return policy_logits, baseline


class Agent:
    """A simple, feed-forward agent."""

    def __init__(self, model: nn.Module, discount: float, device: torch.device):
        self._model = model
        self._discount = discount
        self._device = device

    @torch.no_grad()
    def step(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        """Steps on a single observation."""
        observation = torch.Tensor(state.observation).unsqueeze(0).to(self._device)
        logits, _ = self._model(observation)
        logits = torch.squeeze(logits, dim=0)
        action = torch_dist.Categorical(logits=logits).sample((1,)).squeeze()
        return action.cpu(), logits.cpu()

    def loss(self, trajs: Transition):
        """Computes a loss of trajs wrt params."""
        # Re-run the agent over the trajectories.
        obs = torch.Tensor(trajs.state.observation).to(self._device)
        T, B, *_ = obs.shape

        learner_logits, baseline_with_bootstrap = self._model(obs.flatten(0, 1))
        learner_logits = learner_logits.view(T, B, -1)
        learner_logits = learner_logits[:-1]

        baseline_with_bootstrap = baseline_with_bootstrap.view(T, B)
        # Separate the bootstrap from the value estimates.
        baseline = baseline_with_bootstrap[:-1]
        baseline_tp1 = baseline_with_bootstrap[1:]

        # Remove bootstrap timestep from non-observations.
        _, actions, behavior_logits = tree.map_structure(lambda t: torch.tensor(t[:-1]).to(self._device), trajs)

        # Shift step_type/reward/discount back by one, so that actions match the
        # timesteps caused by the action.
        timestep = tree.map_structure(lambda t: t[1:], trajs.state)
        not_done = torch.Tensor(timestep.discount).to(self._device)
        discount = not_done * self._discount
        # The step is uninteresting if we transitioned LAST -> FIRST.
        mask = torch.roll(not_done, 1, dims=0)
        reward = torch.Tensor(timestep.reward).to(self._device)

        learner_policy = torch_dist.Categorical(logits=learner_logits)
        learner_logits = learner_policy.log_prob(actions)
        behavior_logits = torch_dist.Categorical(logits=behavior_logits).log_prob(actions)
        rho = torch.exp(learner_logits - behavior_logits).detach()
        v_trace_fn = vmap(vtrace.vtrace_td_error_advantage, in_dims=1, out_dims=1)
        adv, td = v_trace_fn(baseline, baseline_tp1, reward, discount, rho)

        # Note that we use mean here, rather than sum as in canonical IMPALA.
        # Compute policy gradient loss.
        pg_loss = - (learner_logits * adv * mask).sum(0).mean()

        # Baseline loss.
        bl_loss = 0.5 * (torch.square(td) * mask).sum(0).mean()
        ent_loss = (learner_policy.entropy() * mask).sum(0).mean()

        actor_loss = pg_loss + 0.01 * ent_loss

        return (actor_loss, bl_loss), {
            "pg_loss": float(pg_loss),
            "bl_loss": float(bl_loss),
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
        torch.nn.utils.clip_grad_value_(self._agent._model.baseline.parameters(), clip)  # noqa
        self._critic_opt.step()
        return extra


def run_actor(agent: Agent, get_params: Callable[[], OrderedDict[str, torch.Tensor]],
              enqueue_traj: Callable[[State], None],
              enqueue_info: Callable[[dict], None],
              unroll_len: int,
              num_trajectories: int):
    """Runs an actor to produce num_trajectories trajectories."""
    env = gym.make(config.env_id)
    env.seed(config.seed)
    obs = env.reset()
    state = State(reward=0., discount=1., observation=obs)

    traj = []
    cumulative_reward = 0.
    for i in range(num_trajectories):
        agent._model.load_state_dict(get_params())  # noqa
        # The first rollout is one step longer.
        for _ in range(unroll_len + int(i == 0)):
            action, logits = agent.step(state)
            transition = Transition(state, action, logits)
            traj.append(transition)
            next_state, reward, done, _ = env.step(action.numpy())

            cumulative_reward += reward
            state = State(reward=reward, observation=next_state, discount=(1 - done) * 1.)
            if done:
                try:
                    enqueue_info({"train/return": cumulative_reward}, block=False)
                except queue.Full:
                    pass
                obs = env.reset()
                state = State(reward=0., discount=1., observation=obs)
                cumulative_reward = 0

        # Stack and send the trajectory.
        stacked_traj = tree.map_structure(lambda *ts: np.stack(ts), *traj)
        enqueue_traj(stacked_traj)
        # Reset the trajectory, keeping the last timestep.
        traj = traj[-1:]


def run(writer: experiment_buddy.WandbWrapper):
    """Runs the example."""

    # Construct the agent network. We need a sample environment for its spec.
    env = gym.make(config.env_id)
    num_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)

    net = SimpleNet(obs_dim, num_actions, config.h_dim).to(config.device)
    behavior = Agent(net, config.gamma, config.device)

    # Construct the agent and learner.
    net = SimpleNet(obs_dim, num_actions, config.h_dim).to(config.device)
    agent = Agent(net, config.gamma, config.device)
    actor_opt = optim.Adam(net.policy.parameters(), config.actor_lr)
    critic_opt = optim.Adam(net.baseline.parameters(), config.critic_lr)
    learner = Learner(agent, actor_opt, critic_opt)

    # Create accessor and queueing functions.
    current_params = lambda: net.state_dict()
    q = queue.Queue(maxsize=config.batch_size)
    q_info = queue.Queue(maxsize=10)

    def dequeue():
        batch = []
        for _ in range(config.batch_size):
            batch.append(q.get())
        batch = tree.map_structure(lambda *ts: np.stack(ts, axis=1), *batch)
        try:
            info = q_info.get(block=False)
        except queue.Empty:
            info = {}
        return batch, info

    # Start the actors.
    for step in range(config.num_actors):
        args = (behavior, current_params, q.put, q_info.put, config.unroll_len, config.trajectories_per_actor)
        threading.Thread(target=run_actor, args=args).start()

    # Run the learner.
    num_steps = config.num_actors * config.trajectories_per_actor // config.batch_size
    for step in trange(num_steps):
        traj, infos = dequeue()
        extra = learner.update(traj)
        for tag, value in {**extra, **infos}.items():
            writer.add_scalar(tag, value, global_step=step)


class config:
    device = torch.device("cuda")
    env_id = "CartPole-v0"
    trajectories_per_actor = 5000
    num_actors = 4
    batch_size = 2
    unroll_len = 20
    gamma = 0.99
    h_dim = 32
    actor_lr = 5e-4
    critic_lr = 5e-3
    seed = 0


def main():
    experiment_buddy.register_defaults(dict(vars(config)))
    writer = experiment_buddy.deploy(host="mila", sweep_yaml="sweep.yaml", proc_num=5)
    run(writer)


if __name__ == '__main__':
    main()
