import contextlib
import logging
import os
import random
import time
from typing import Tuple, Union

import gym
import numpy as np
import torch

T = torch.Tensor


def get_logger(name):
    os.makedirs("logs", exist_ok=True)
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join("logs", name)))
    return logger


class GymWrapper(gym.Wrapper):
    def step(self, action):
        if self.should_reset:
            return self.reset()
        s, r, d, info = super(GymWrapper, self).step(action)
        self.cumulative_reward += r
        if d:
            info = {
                "step": self.t,
                "cumulative_return": self.cumulative_reward

            }
            self.should_reset = True
        self.t += 1
        return torch.from_numpy(s), torch.tensor(r, dtype=torch.float32), torch.tensor(d, dtype=torch.float32), info

    def reset(self, **kwargs):
        s = super(GymWrapper, self).reset()
        self.t = 0
        self.cumulative_reward = 0
        self.should_reset = False
        return torch.from_numpy(s), torch.tensor(0.), torch.tensor(0.), {}


@contextlib.contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: T) -> None:
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: T, batch_var: T, batch_count: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
