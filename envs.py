from typing import NamedTuple

import envpool
import gym
import numpy as np
import procgen
import rlmeta.envs.env
import torch
from rlmeta.core.types import TimeStep, Action
from rlmeta.envs.gym_wrappers import GymWrapper


# TODO: create environment spec type

class EnvPool(GymWrapper):
    def step(self, action: Action) -> TimeStep:
        # The new gym api carries terminal. We don't care for now, as evnpool, skip one step which we mask later
        # in the env
        s, r, d, terminated, info = self.env.step(action.action.cpu().numpy())
        r = torch.tensor(r)
        d = torch.tensor(d, dtype=torch.bool)
        info["terminated"] = torch.tensor(terminated, dtype=torch.bool)
        return TimeStep(observation=self._observation_fn(s), reward=r, done=d, info=info)

    def reset(self, *args, **kwargs) -> TimeStep:
        s, info = self.env.reset(*args, **kwargs)
        return TimeStep(observation=self._observation_fn(s), reward=0, done=False, info=info)


def control_obs_fn(obs):
    return torch.from_numpy(obs).to(torch.float32)


def squeze_obs(obs):
    return torch.from_numpy(obs).squeeze(0)


class EnvSpec(NamedTuple):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space


class EnvFactory(rlmeta.envs.env.EnvFactory):
    def __init__(self, task_id: str, library_str: str = 'atari', train: bool = True, seed: int = 33):
        self.task_id = task_id
        self.library = library_str
        self.train = train
        self.seed = seed
        if library_str == 'mujoco':
            self.observation_fn = control_obs_fn
        else:
            self.observation_fn = squeze_obs

    def __call__(self, seed) -> EnvPool:
        make_fn = _libraries[self.library]
        return EnvPool(make_fn(self.task_id, seed=seed % self.seed, train=self.train),
                       observation_fn=self.observation_fn)

    def get_spec(self):
        env = self.__call__(0)
        env_spec = EnvSpec(env.observation_space, env.action_space)
        del env
        return env_spec


class ProcWrap(gym.Wrapper):
    render_mode = "rgb_array"

    def __init__(self, env: procgen.ProcgenEnv, train: bool = True):
        super().__init__(env)
        observation_space = env.observation_space['rgb']
        h, w, c = observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)
        self.reward_range = (-10, 10)
        self.done = None
        if not train:
            self._reward_fn = lambda x: x

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = {k: [d[k] for d in info] for k in info[0]}
        return self._observation_fn(obs), self._reward_fn(rew), done, done, info

    def seed(self, *args, **kwargs):
        return

    def reset(self, **kwargs):
        if self.done is not None and not self.done:
            print("WARNING: reseting env before done")
        obs = self._observation_fn(self.env.reset(**kwargs))
        return obs, {}

    def _observation_fn(self, obs):
        return obs['rgb'].transpose(0, 3, 1, 2).squeeze(0)

    def _reward_fn(self, r):
        return np.clip(r, *self.reward_range)


def make_procgen(task_id='starpilot', num_envs=1, seed=33, train: bool = True):
    env = procgen.ProcgenEnv(num_envs=num_envs, env_name=task_id, rand_seed=seed, num_levels=0, start_level=0,
                             distribution_mode="easy")

    env = ProcWrap(env, train)

    return env


def make_atari(task_id, batch_size=1, seed=33, async_envs=False, train: bool = True):
    num_envs = batch_size
    if async_envs:
        num_envs = batch_size * 3
    reward_clip = False
    episodic_life = False
    if train:
        reward_clip = True
        episodic_life = True
    env = envpool.make_gym(task_id, batch_size=batch_size, num_envs=num_envs, seed=seed, reward_clip=reward_clip,
                           # taken from dqn zoo
                           episodic_life=episodic_life, max_episode_steps=10800)

    return env


def make_mujoco(task_id, batch_size=1, seed=33, train: bool = True):
    num_envs = batch_size
    env = envpool.make_gym(task_id, batch_size=batch_size, num_envs=num_envs, seed=seed)
    return env


_libraries = {"procgen": make_procgen, "atari": make_atari, "mujoco": make_mujoco}
