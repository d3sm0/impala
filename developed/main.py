# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses

import experiment_buddy
import torch.multiprocessing as mp
import torch.optim
from rlmeta.core.controller import Controller
from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.samplers import UniformSampler
from rlmeta.storage import TensorCircularBuffer

import envs
import ppo
import utils
import wandb
from configs.config import Config
from models.distributed_models import AtariPPOModel
from ppo import PPOAgent

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# @hydra.main(config_path="configs", config_name="config")
def main(cfg: Config):
    # config_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_buddy.register_defaults(dataclasses.asdict(cfg))
    writer = experiment_buddy.deploy(host="", wandb_kwargs={"project": "impala",
                                                                "settings": wandb.Settings(start_method="thread"),
                                                                "tags": [cfg.task.benchmark]
                                                                },
                                     extra_modules=["cuda/11.1/cudnn/8.0", "python/3.7", "gcc", "libffi"])

    # writer = wandb.init(project="impala",
    #                     name=f"{cfg.task.env_id}-{cfg.task.benchmark}",
    #                     mode="disabled",
    #                     config=dataclasses.asdict(cfg),
    #                     settings=wandb.Settings(start_method="thread"))
    # TODO: make me ready for distributed training
    env = envs.EnvFactory(cfg.task.env_id, cfg.task.benchmark)(0)
    train_model = AtariPPOModel(cfg.task.benchmark, env.observation_space.shape, env.action_space.n).to(
        cfg.distributed.train_device)

    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.optimizer.lr, eps=cfg.optimizer.eps)
    infer_model = copy.deepcopy(train_model).to(cfg.distributed.infer_device).eval()

    ctrl = Controller()
    rb = ReplayBuffer(TensorCircularBuffer(cfg.agent.replay_buffer_size),
                      UniformSampler())

    servers = utils.create_servers(cfg, ctrl, infer_model, rb)
    # TODO: can we make the infer model optimized for inference?
    async_model, async_ctrl, async_rb = utils.create_master(cfg, ctrl, train_model, rb)

    Agent = ppo.agents_registry[cfg.agent.name]

    loops = utils.create_workers(cfg, ctrl, PPOAgent, infer_model, rb)

    agent = Agent(async_model,
                  replay_buffer=async_rb,
                  controller=async_ctrl,
                  optimizer=optimizer)

    servers.start()
    loops.start()
    agent.connect()

    for epoch in range(cfg.training.num_epochs):
        stats = agent.eval(cfg.evaluation.num_epsisodes, keep_training_loops=True).dict()
        for k, v in stats.items():
            writer.run.log({f"eval/{k}": v['mean']})
        train_stats = agent.train(cfg.training.steps_per_epoch, writer).dict()
        avg_n_samples = train_stats['episode_length']['mean'] * train_stats['episode_length']['count']
        if avg_n_samples > cfg.task.total_frames:
            break
    loops.terminate()
    servers.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(Config())
