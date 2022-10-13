from typing import NamedTuple

import envpool
import gym
import numpy as np
import procgen
import rlmeta.envs.env
import torch
from gym.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from rlmeta.core.types import TimeStep, Action
from rlmeta.envs.gym_wrappers import GymWrapper


# TODO: create environment spec type

class EnvPool(GymWrapper):
    def step(self, action: Action) -> TimeStep:
        # The new gym api carries terminal. We don't care for now, as evnpool, skip one step which we mask later
        # in the env
        s, r, d, _, info = self.env.step(action.action.cpu().numpy())
        return TimeStep(observation=self._observation_fn(s), reward=torch.tensor(r),
                        done=torch.tensor(d, dtype=torch.bool), info=info)

    def reset(self, *args, **kwargs) -> TimeStep:
        s = self.env.reset(*args, **kwargs)
        info = {}
        if isinstance(s, tuple):
            s, info = s
        return TimeStep(observation=self._observation_fn(s), reward=0, done=False, info=info)


def squeze_obs(obs):
    return torch.from_numpy(obs).squeeze(0)


class EnvSpec(NamedTuple):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space


class EnvFactory(rlmeta.envs.env.EnvFactory):
    def __init__(self, task_id: str, library_str: str = 'atari'):
        self.task_id = task_id
        self.library = library_str

    def __call__(self, seed) -> EnvPool:
        make_fn = _libraries[self.library]
        return EnvPool(make_fn(self.task_id, seed=seed), observation_fn=squeze_obs)

    def get_spec(self):
        env = self.__call__(0)
        env_spec = EnvSpec(env.observation_space, env.action_space)
        del env
        return env_spec


class ProcWrap(gym.Wrapper):
    render_mode = "rgb_array"

    def __init__(self, env: procgen.ProcgenEnv):
        super().__init__(env)
        self.observation_space = env.observation_space['rgb']
        self.reward_range = (-10, 10)
        self.is_vector_env = True
        self.num_envs = env.num_envs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info = {k: [d[k] for d in info] for k in info[0]}
        return self._observation_fn(obs), self._reward_fn(rew), done, done, info

    def seed(self, *args, **kwargs):
        return

    def reset(self, **kwargs):
        return self._observation_fn(self.env.reset(**kwargs))

    def _observation_fn(self, obs):
        return obs['rgb'].transpose(0, 3, 1, 2).squeeze(0)

    def _reward_fn(self, r):
        return np.clip(r, -10, 10)


class AtariWrap(gym.Wrapper):
    def __init__(self, env):
        super(AtariWrap, self).__init__(env)

    def step(self, action):
        step_returns = self.env.step(action)
        return convert_to_terminated_truncated_step_api(step_returns, True)


def make_procgen(task_id='starpilot', num_envs=1, seed=33):
    env = procgen.ProcgenEnv(num_envs=num_envs, env_name=task_id, rand_seed=seed, num_levels=0, start_level=0,
                             distribution_mode="easy")

    env = ProcWrap(env)
    env = gym.wrappers.StepAPICompatibility(env, True)
    # env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # envs = gym.wrappers.TransformObservation(envs, lambda x: (x['rgb']).transpose(0, 3, 1, 2))
    # envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return env


def make_atari(task_id, batch_size=1, seed=33, async_envs=False):
    num_envs = batch_size
    if async_envs:
        num_envs = batch_size * 3
    env = envpool.make_gym(task_id, batch_size=batch_size, num_envs=num_envs, seed=seed)
    env.is_vector_env = True
    env.num_envs = batch_size
    env = AtariWrap(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


_libraries = {"procgen": make_procgen, "atari": make_atari}
