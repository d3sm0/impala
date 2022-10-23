# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import random
import sys
import time

import experiment_buddy
import hydra
import moolib
import numpy as np

moolib.set_log_level("debug")
import omegaconf
import rlmeta.utils.hydra_utils as hydra_utils
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from rlmeta.core.controller import Controller
from rlmeta.core.loop import LoopList

import envs
import utils
import wandb
from agents.distributed_agent import DistributedAgent
from agents.sac.builder import SACBuilder

logger = logging.getLogger("root")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


# from agents.ppo.builder import PPOBuilder
# from agents.impala.builder import ImpalaBuilder


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg):
    logger.info(hydra_utils.config_to_json(cfg))

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    # Meh
    if "SLURM_JOB_ID" in os.environ.keys() or cfg.distributed.host == "mila":
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.load("conf/deploy/mila.yaml"))
    else:
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.load("conf/deploy/local.yaml"))

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
    # writer = utils.Writer()
    # builder = ApexDQNBuilder(cfg)
    # builder = ApexDistributionalBuilder(cfg)
    # builder = PPOBuilder(cfg)
    # builder = ImpalaBuilder(cfg)
    builder = SACBuilder(cfg)

    env_factory = envs.EnvFactory(cfg.task.env_id, library_str=cfg.task.benchmark)
    # TODO: make spec here
    train_model = builder.make_network(env_factory.get_spec())
    rb = builder.make_replay()

    ctrl = Controller()
    servers = utils.create_servers(cfg, ctrl, builder.actor_model, rb)
    t_ctrl, t_model, t_rb = utils.create_workers(cfg, ctrl, builder.actor_model, rb)

    t_agent_fac = builder.make_actor(t_model, t_rb, deterministic=False)
    train_loop = utils.create_train_loop(cfg, env_factory, t_agent_fac, t_ctrl)
    e_ctrl, e_model, _ = utils.create_workers(cfg, ctrl, builder.actor_model)

    e_agent_fac = builder.make_actor(e_model, deterministic=True)
    evaluate_loop = utils.create_evaluation_loops(cfg, envs.EnvFactory(cfg.task.env_id, library_str=cfg.task.benchmark,
                                                                       train=False), e_agent_fac,
                                                  e_ctrl)

    loops = LoopList([train_loop, evaluate_loop])

    a_model, a_ctrl, a_rb = utils.create_master(cfg, ctrl, train_model, rb)

    learner = builder.make_learner(a_model, a_rb)
    agent = DistributedAgent(a_ctrl, learner, writer)

    servers.start()
    loops.start()
    agent.connect()
    for epoch in range(cfg.training.num_epochs):
        total_samples = None
        for t in range(7):
            try:
                total_samples = agent.train(cfg.training.steps_per_epoch)
                break
            except RuntimeError as e:
                print(f"RuntimeError. Sleeping for {2 ** t}", e)
                time.sleep(2 ** t)
        if total_samples is None:
            print("Failed to train")
            break
        agent.eval(cfg.evaluation.num_rollouts, keep_training_loops=False)
        if total_samples > cfg.task.total_frames:
            break
    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
