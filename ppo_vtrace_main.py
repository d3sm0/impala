# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os

import hydra
import rlmeta.utils.hydra_utils as hydra_utils
import torch.multiprocessing as mp
import torch.optim

import envs
import wandb
from models import AtariPPOModel
from ppo import ImpalaAgent
from utils import prepare_distributed

os.environ["OMP_NUM_THREADS"] = "1"


@hydra.main(config_path="./conf", config_name="conf_impala")
def main(cfg):
    logging.info(hydra_utils.config_to_json(cfg))
    # we should have a function that takes the env and algos and return the model

    env = envs.make_atari(cfg.task.env_id)
    train_model = AtariPPOModel(cfg.task.benchmark, env.observation_space.shape, env.action_space.n).to(
        cfg.distributed.train_device)

    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    infer_model = copy.deepcopy(train_model).to(cfg.distributed.infer_device)
    servers, loops, async_model, async_ctrl, async_rb = prepare_distributed(cfg, ImpalaAgent, infer_model, train_model)

    agent = ImpalaAgent(async_model,
                        replay_buffer=async_rb,
                        controller=async_ctrl,
                        optimizer=optimizer,
                        batch_size=cfg.agent.batch_size,
                        learning_starts=cfg.agent.learning_starts,
                        model_push_period=cfg.agent.model_push_period,
                        max_grad_norm=cfg.agent.max_grad_norm,
                        entropy_coeff=cfg.agent.entropy_cost,
                        rollout_length=cfg.agent.rollout_length
                        )

    servers.start()
    loops.start()
    agent.connect()

    writer = wandb.init(project="impala")
    for epoch in range(cfg.training.num_epochs):
        # # stats = agent.eval(cfg.num_eval_episodes, keep_training_loops=True)
        # for k, v in stats.dict().items():
        #     writer.log({f"{k}": v['mean']})
        stats = agent.train(cfg.training.steps_per_epoch, writer)
        # torch.save(train_model.state_dict(), f"ppo_agent-{epoch}.pth")
    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
