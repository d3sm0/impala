import contextlib
import logging
import os
import random
import time

import gym
import numpy as np
import torch


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
