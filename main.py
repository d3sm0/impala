# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import experiment_buddy
import hydra
import torch.multiprocessing as mp
import torch.optim
from omegaconf import OmegaConf

import envs
import wandb
from configs.config import Config
from models.distributed_models import AtariPPOModel
from ppo import PPOAgent
from utils import prepare_distributed

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


@hydra.main(config_path="configs", config_name="config")
def main(cfg: Config):
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_buddy.register(config_dict)
    writer = experiment_buddy.deploy(host="mila", wandb_kwargs={"project": "impala",
                                                                "settings": wandb.Settings(start_method="thread")})

    # writer = wandb.init(project="impala", mode="disabled", config=config_dict,
    #                     settings=wandb.Settings(start_method="thread"))

    # we should have a function that takes the env and algos and return the model

    env = envs.EnvFactory(cfg.task.env_id, cfg.task.benchmark)(0)
    train_model = AtariPPOModel(cfg.task.benchmark, env.observation_space.shape, env.action_space.n).to(
        cfg.distributed.train_device)

    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    infer_model = copy.deepcopy(train_model).to(cfg.distributed.infer_device)
    servers, loops, async_model, async_ctrl, async_rb = prepare_distributed(cfg, PPOAgent, infer_model, train_model)

    agent = PPOAgent(async_model,
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

    for epoch in range(cfg.training.num_epochs):
        # stats = agent.eval(cfg.num_eval_episodes, keep_training_loops=True).dict()
        # stats = agent._controller.stats(phase=Phase.EVAL).dict()
        # if not len(stats):
        #     continue

        # writer.log(stats)
        stats = agent.train(cfg.training.steps_per_epoch, writer)
        # torch.save(train_model.state_dict(), f"ppo_agent-{epoch}.pth")
    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
