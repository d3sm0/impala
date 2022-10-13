import abc
from typing import Optional

from rlmeta.core import remote
from rlmeta.core.model import ModelLike
from rlmeta.core.replay_buffer import ReplayBufferLike
from rlmeta.core.types import TimeStep, Action

import envs


class Agent(abc.ABC):

    @abc.abstractmethod
    def train(self, num_steps: int) -> None:
        ...

    @abc.abstractmethod
    def eval(self, num_episodes: int, keep_training_loops: bool) -> None:
        ...


class Actor(abc.ABC):

    def connect(self) -> None:
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, remote.Remote):
                obj.connect()

    @abc.abstractmethod
    async def async_act(self, timestep: TimeStep) -> Action:
        ...

    @abc.abstractmethod
    async def async_observe_init(self, timestep: TimeStep) -> None:
        ...

    @abc.abstractmethod
    async def async_observe(self, action: Action,
                            next_timestep: TimeStep) -> None:
        ...

    @abc.abstractmethod
    async def async_update(self) -> None:
        ...


class Learner(abc.ABC):
    _step_counter: int
    can_train: bool = False

    @abc.abstractmethod
    def train_step(self):
        ...

    @abc.abstractmethod
    def prepare(self):
        ...

    def connect(self) -> None:
        for obj_name in dir(self):
            obj = getattr(self, obj_name)
            if isinstance(obj, remote.Remote):
                obj.connect()


class Builder(abc.ABC):
    @abc.abstractmethod
    def make_replay(self):
        ...

    @abc.abstractmethod
    def make_actor(self, model: ModelLike, rb: Optional[ReplayBufferLike] = None, deterministic: bool = False):
        ...

    @abc.abstractmethod
    def make_learner(self, model: ModelLike, rb: ReplayBufferLike):
        ...

    @abc.abstractmethod
    def make_network(self, env_spec: envs.EnvSpec):
        ...

    @property
    def actor_model(self):
        return getattr(self, "_actor_model", None)

    @property
    def learner_model(self):
        return getattr(self, "_learner_model", None)
