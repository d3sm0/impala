# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Optional

import rich.progress
from rlmeta.core.controller import Phase, ControllerLike
from rlmeta.utils.stats_dict import StatsDict

from agents.core import Agent, Learner


class DistributedAgent(Agent):
    def __init__(self, controller: ControllerLike, learner: Learner, writer=None):
        self._controller = controller
        self._learner = learner
        self._writer = writer
        self._stats_dict = StatsDict()
        self._start_time = time.perf_counter()

    def set_phase(self, phase=Phase.TRAIN):
        self._controller.set_phase(phase=phase)

    def train(self, num_steps: int) -> int:
        self._controller.set_phase(Phase.TRAIN)
        self._learner.prepare()
        for local_steps in rich.progress.track(range(num_steps), description="Training"):
            metrics = self._learner.train_step()
            self._stats_dict.extend(metrics)
            if local_steps % 100 == 0:
                # self._writer.run.log({f"{k}_mean": v['mean'] for k, v in self._stats_dict.dict().items() if "debug" in k})
                self._writer.run.log(metrics)
        remote_metrics = self._controller.stats(Phase.TRAIN).dict()
        total_samples = remote_metrics["episode_length"]["mean"] * remote_metrics["episode_length"]["count"]
        delta_samples = (total_samples / (time.perf_counter() - self._start_time))
        self._writer.run.log({f"train_envs/{k.replace('/', '_')}": v['mean'] for k, v in remote_metrics.items()})
        self._writer.run.log({"debug/samples_per_second": delta_samples})
        return total_samples

    def eval(self,
             num_episodes: Optional[int] = None,
             keep_training_loops: bool = True) -> Optional[StatsDict]:
        if keep_training_loops:
            self._controller.set_phase(Phase.BOTH)
        else:
            self._controller.set_phase(Phase.EVAL)
        self._controller.reset_phase(Phase.EVAL, limit=num_episodes)
        while self._controller.count(Phase.EVAL) < num_episodes:
            time.sleep(1)
        stats = self._controller.stats(Phase.EVAL)
        self._writer.run.log({f"eval_envs/{k.replace('/', '_')}": v['mean'] for k, v in stats.dict().items()})
        return stats

    def connect(self):
        self._controller.connect()
        self._learner.connect()
