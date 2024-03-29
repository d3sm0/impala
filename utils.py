import contextlib
import random
import time
from typing import Optional

from rlmeta.utils import nested_utils

try:
    import aim
except ImportError:
    print(f"aim not available")
    pass
import numpy as np
import torch
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.controller import Controller, Phase
from rlmeta.core.loop import ParallelLoop
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.remote import Remote
from rlmeta.core.replay_buffer import ReplayBuffer, RemoteReplayBuffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.types import Action, TimeStep
from rlmeta.envs.env import EnvFactory

TIMEOUT = 240


@contextlib.contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


class FakeRun:
    def track(self, *args, **kwargs):
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Run:
    def __init__(self, disabled=False):
        if disabled:
            self.run = FakeRun()
        else:
            self.run = aim.Run()
        self.config = {}

    def save(self, *args, **kwargs):
        pass

    def log(self, metrics, step=None):
        for k, v in metrics.items():
            self.run.track(v, k, step=step)

    def add_figures(self, figures, step=None):
        for k, v in figures.items():
            self.run.track(aim.Figure(v), k, step=step)


class Writer:
    def __init__(self, disabled=False):
        self.run = Run(disabled)


class RecordMetrics(EpisodeCallbacks):
    def on_episode_step(self, index: int, step: int, action: Action, timestep: TimeStep) -> None:
        if "episode" in timestep.info.keys():
            results = timestep.info["episode"]
            mask = timestep.info["_episode"]
            for key, value in results.items():
                self.custom_metrics[key] = (value * mask).sum() / mask.sum()
            # self.custom_metrics["sps"] = (self.custom_metrics["l"] / self.custom_metrics["t"])


def create_master(cfg, ctrl: Controller, train_model: RemotableModel, rb: ReplayBuffer):
    assert train_model.training is True
    a_rb = RemoteReplayBuffer(rb, cfg.distributed.r_server_name, cfg.distributed.r_server_addr, timeout=TIMEOUT,
                              prefetch=cfg.training.prefetch)
    async_ctrl = Remote(ctrl, cfg.distributed.c_server_name, cfg.distributed.c_server_addr, timeout=TIMEOUT)
    async_model = DownstreamModel(train_model, cfg.distributed.m_server_name, cfg.distributed.m_server_addr,
                                  timeout=TIMEOUT)
    return async_model, async_ctrl, a_rb


def create_servers(cfg, ctrl: Controller, model: RemotableModel, rb: ReplayBuffer):
    # there is only one master
    model_server = Server(cfg.distributed.m_server_name, cfg.distributed.m_server_addr)
    rb_server = Server(cfg.distributed.r_server_name, cfg.distributed.r_server_addr)
    controller_server = Server(cfg.distributed.c_server_name, cfg.distributed.c_server_addr)

    model_server.add_service(model)
    rb_server.add_service(rb)
    controller_server.add_service(ctrl)
    servers = ServerList([model_server, rb_server, controller_server])
    return servers


def create_workers(cfg, ctrl: Controller, infer_model: RemotableModel, rb: Optional[ReplayBuffer] = None):
    # this actions happens in the worker machine
    infer_model = Remote(infer_model, cfg.distributed.m_server_name, cfg.distributed.m_server_addr, timeout=TIMEOUT)
    train_ctrl = Remote(ctrl, cfg.distributed.c_server_name, cfg.distributed.c_server_addr, timeout=TIMEOUT)
    train_rb = None
    if rb is not None:
        train_rb = RemoteReplayBuffer(rb, cfg.distributed.r_server_name, cfg.distributed.r_server_addr, timeout=TIMEOUT,
                                      prefetch=cfg.training.prefetch)
    return train_ctrl, infer_model, train_rb


def create_train_loop(cfg, env_fac: EnvFactory, agent_factory: AgentFactory, train_ctrl: Controller):
    train_loop = ParallelLoop(env_fac,
                              agent_factory,
                              train_ctrl,
                              running_phase=Phase.TRAIN,
                              should_update=True,
                              num_workers=cfg.training.num_workers,
                              num_rollouts=cfg.training.num_rollouts,
                              seed=cfg.training.seed,
                              # episode_callbacks=RecordMetrics(),
                              )
    return train_loop


def create_evaluation_loops(cfg, env_factory: EnvFactory, agent_factory: AgentFactory,
                            evaluate_ctrl: Controller):
    evaluate_loop = ParallelLoop(env_factory,
                                 agent_factory,
                                 evaluate_ctrl,
                                 running_phase=Phase.EVAL,
                                 should_update=False,
                                 num_rollouts=cfg.evaluation.num_workers,
                                 num_workers=cfg.evaluation.num_rollouts,
                                 seed=cfg.evaluation.seed,
                                 # episode_callbacks=RecordMetrics(),
                                 )
    return evaluate_loop


class MockBatcher:
    def __init__(self, size, device, dim):
        self._size = size
        self._device = device
        self._dim = dim
        self._batcher = []
        self._stack_tensors = False

    def get(self):
        fn = torch.stack if self._stack_tensors else torch.cat
        batched_data = nested_utils.collate_nested(fn, self._batcher)
        self._batcher.clear()
        return batched_data

    def cat(self, *args):
        self._batcher.append(args)

    def stack(self, *args):
        self._batcher.append(args)
        self._stack_tensors = True

    def size(self):
        return len(self._batcher)

    def empty(self):
        return len(self._batcher) < self._size
