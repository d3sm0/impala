# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys
import time

import experiment_buddy
import hydra
import omegaconf
import rlmeta.utils.hydra_utils as hydra_utils
import torch.multiprocessing as mp
import wandb
from rlmeta.core.controller import Controller
from rlmeta.core.loop import LoopList

import envs
import utils
from agents.distributed_agent import DistributedAgent
# from agents.dqn.builder import ApexDQNBuilder
from agents.ppo.builder import PPOBuilder


# from agents.impala.builder import ImpalaBuilder


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    logging.info(hydra_utils.config_to_json(cfg))

    writer = experiment_buddy.deploy(host=cfg.distributed.host,
                                     disabled=sys.gettrace() is not None,
                                     wandb_kwargs={"project": "impala",
                                                   "settings": wandb.Settings(start_method="thread"),
                                                   "config": omegaconf.OmegaConf.to_container(
                                                       cfg, resolve=True),
                                                   # "tags": [cfg.distributed.host,
                                                   #          cfg.task.benchmark]
                                                   },
                                     extra_modules=["cuda/11.1/cudnn/8.0", "python/3.7", "gcc", "libffi"])
    builder = PPOBuilder(cfg)

    env_factory = envs.EnvFactory(cfg.task.env_id)
    # TODO: make spec here
    train_model = builder.make_network(env_factory.get_spec())
    rb = builder.make_replay()

    ctrl = Controller()
    servers = utils.create_servers(cfg, ctrl, builder.actor_model, rb)
    e_ctrl, e_model, t_ctrl, t_model, t_rb = utils.create_workers(cfg, ctrl, builder.actor_model, rb)

    t_agent_fac = builder.make_actor(t_model, t_rb, deterministic=False)
    train_loop = utils.create_train_loop(cfg, env_factory, t_agent_fac, t_ctrl)

    e_agent_fac = builder.make_actor(e_model, deterministic=True)
    evaluate_loop = utils.create_evaluation_loops(cfg, env_factory, e_agent_fac, e_ctrl)

    loops = LoopList([train_loop, evaluate_loop])

    a_model, a_ctrl, a_rb = utils.create_master(cfg, ctrl, train_model, rb)

    learner = builder.make_learner(a_model, a_rb)
    agent = DistributedAgent(a_ctrl, learner, writer)

    servers.start()
    loops.start()
    agent.connect()

    for epoch in range(cfg.training.num_epochs):
        agent.eval(cfg.evaluation.num_rollouts, keep_training_loops=True)
        agent.train(cfg.training.steps_per_epoch)
    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
