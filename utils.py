import contextlib
import random
import time

# import aim
import numpy as np
import torch
from rlmeta.agents.agent import AgentFactory
from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.controller import Controller, Phase
from rlmeta.core.loop import ParallelLoop, LoopList
from rlmeta.core.model import DownstreamModel, RemotableModel
from rlmeta.core.remote import Remote
from rlmeta.core.replay_buffer import ReplayBuffer, RemoteReplayBuffer
from rlmeta.core.server import Server, ServerList
from rlmeta.core.types import Action, TimeStep
from rlmeta.samplers import UniformSampler
from rlmeta.storage import TensorCircularBuffer

import envs
from configs.config import Config

TIMEOUT = 60


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

    def log(self, metrics, step):
        for k, v in metrics.items():
            self.run.track(v, k, step=step)

    def add_figures(self, figures, step):
        for k, v in figures.items():
            self.run.track(aim.Figure(v), k, step=step)

    def __del__(self):
        self.run.finalize()


class Writer:
    def __init__(self, disabled):
        self.run = Run(disabled)

    def __del__(self):
        del self.run


class RecordMetrics(EpisodeCallbacks):
    def on_episode_step(self, index: int, step: int, action: Action, timestep: TimeStep) -> None:
        if "episode" in timestep.info.keys():
            results = timestep.info["episode"]
            mask = timestep.info["_episode"]
            for key, value in results.items():
                self.custom_metrics[key] = (value * mask).sum() / mask.sum()
            self.custom_metrics["sps"] = (self.custom_metrics["l"] / self.custom_metrics["t"])


def prepare_distributed(cfg: Config, RemoteAgent, infer_model: RemotableModel, train_model: RemotableModel):
    ctrl = Controller()

    rb = ReplayBuffer(TensorCircularBuffer(cfg.agent.replay_buffer_size),
                      UniformSampler())

    model_server = Server(cfg.distributed.m_server_name, cfg.distributed.m_server_addr)
    rb_server = Server(cfg.distributed.r_server_name, cfg.distributed.r_server_addr)
    controller_server = Server(cfg.distributed.c_server_name, cfg.distributed.c_server_addr)

    model_server.add_service(infer_model)
    rb_server.add_service(rb)
    controller_server.add_service(ctrl)

    servers = ServerList([model_server, rb_server, controller_server])

    async_model = DownstreamModel(train_model, model_server.name, model_server.addr, timeout=TIMEOUT)
    # evaluate_model = Remote(infer_model, model_server.name, model_server.addr, None, timeout=60)
    infer_model = Remote(infer_model, model_server.name, model_server.addr, timeout=TIMEOUT)

    async_ctrl = Remote(ctrl, controller_server.name, controller_server.addr, timeout=TIMEOUT)
    train_ctrl = Remote(ctrl, controller_server.name, controller_server.addr, timeout=TIMEOUT)
    # evaluate_ctrl = Remote(ctrl, controller_server.name, controller_server.addr, None, timeout=60)

    a_rb = RemoteReplayBuffer(rb, rb_server.name, rb_server.addr, timeout=TIMEOUT, prefetch=cfg.agent.prefetch)
    train_rb = RemoteReplayBuffer(rb, rb_server.name, rb_server.addr, timeout=TIMEOUT)
    # TODO just from config env
    env_fac = envs.EnvFactory(cfg.task.env_id, cfg.task.benchmark)
    # TODO: we might need to pass kwargs here
    train_agent_factory = AgentFactory(RemoteAgent, infer_model, replay_buffer=train_rb,
                                       rollout_length=cfg.agent.rollout_length)
    train_loop = ParallelLoop(env_fac,
                              train_agent_factory,
                              train_ctrl,
                              running_phase=Phase.TRAIN,
                              should_update=True,
                              num_workers=cfg.training.num_workers,
                              num_rollouts=cfg.training.num_rollouts,
                              seed=cfg.training.seed,
                              episode_callbacks=RecordMetrics(),
                              )
    # evaluate_agent_fac = AgentFactory(RemoteAgent, evaluate_model, deterministic_policy=cfg.deterministic_policy)
    # evaluate_loop = ParallelLoop(env_fac,
    #                              evaluate_agent_fac,
    #                              evaluate_ctrl,
    #                              running_phase=Phase.EVAL,
    #                              should_update=False,
    #                              num_rollouts=cfg.num_eval_rollouts,
    #                              num_workers=cfg.num_eval_workers,
    #                              seed=cfg.eval_seed,
    #                              episode_callbacks=RecordMetrics(),
    #                              )
    loops = LoopList([train_loop])  # , evaluate_loop])
    return servers, loops, async_model, async_ctrl, a_rb
