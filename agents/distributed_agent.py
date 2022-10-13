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
        self.start_time = time.perf_counter()

    def train(self, num_steps: int) -> None:
        self._controller.set_phase(Phase.TRAIN)
        self._learner.prepare()
        for local_steps in rich.progress.track(range(num_steps), description="Training"):
            metrics = self._learner.train_step()
            if local_steps % 100 == 0:
                remote_metrics = self._controller.stats(Phase.TRAIN).dict()
                new_samples = remote_metrics["episode_length"]["mean"] * remote_metrics["episode_length"]["count"]
                delta_samples = (new_samples / (time.perf_counter() - self.start_time))
                metrics["debug/samples_per_second"] = delta_samples
                metrics["debug/gradient_per_second"] = self._learner._step_counter / (time.perf_counter() - self.start_time)
                self._writer.run.log(metrics)

        episode_stats = self._controller.stats(Phase.TRAIN)
        self._writer.run.log({f"train_envs/{k.replace('/', '_')}": v['mean'] for k, v in episode_stats.dict().items()})

    def eval(self,
             num_episodes: Optional[int] = None,
             keep_training_loops: bool = True) -> Optional[StatsDict]:
        if keep_training_loops:
            self._controller.set_phase(Phase.BOTH)
        else:
            self._controller.set_phase(Phase.EVAL)
        self._controller.reset_phase(Phase.EVAL, limit=num_episodes)
        while self._controller.count(Phase.EVAL) < num_episodes:
            time.sleep(0.1)
        stats = self._controller.stats(Phase.EVAL)
        self._writer.run.log({f"eval_envs/{k.replace('/', '_')}": v['mean'] for k, v in stats.dict().items()})
        return stats

    def connect(self):
        self._controller.connect()
        self._learner.connect()
