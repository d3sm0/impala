import dataclasses
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class Distributed:
    m_server_name: str = "m_server"
    m_server_addr: str = "127.0.0.1:4411"

    r_server_name: str = "r_server"
    r_server_addr: str = "127.0.0.1:4412"

    c_server_name: str = "c_server"
    c_server_addr: str = "127.0.0.1:4413"

    train_device: str = "cuda:0"
    infer_device: str = "cuda:0"


@dataclasses.dataclass
class Evaluation:
    num_workers: int = 4
    num_rollouts: int = 8
    seed: int = 456
    deterministic_policy: bool = True


@dataclasses.dataclass
class Training:
    num_workers: int = 8
    num_rollouts: int = 64
    seed: int = 123
    num_epochs: int = 3000
    steps_per_epoch: int = 1000


# TODO: how to set variables at runtime?
@dataclasses.dataclass
class Task:
    env_id: str = "starpilot"
    benchmark: str = "procgen"


@dataclasses.dataclass
class Optimizer:
    lr: float
    eps: float


@dataclasses.dataclass
class Adam(Optimizer):
    _target_: str = "torch.optim.Adam"
    lr: float = 2.5e-4
    eps: float = 1e-5


@dataclasses.dataclass
class Agent:
    batch_size: int = MISSING
    learning_starts: Optional[int] = MISSING
    rollout_length: int = MISSING
    replay_buffer_size: int = MISSING
    entropy_cost: float = 0.
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    prefetch: int = 2
    model_push_period: int = 8


@dataclasses.dataclass
class Impala(Agent):
    _target_: str = "impala.ImpalaAgent"
    replay_buffer_size: int = 256
    learning_starts: int = 100
    rollout_length: int = 80
    batch_size: int = 8
    model_push_period: int = 4


@dataclasses.dataclass
class PPO(Agent):
    _target_: str = "ppo.PPOAgent"
    lambda_: float = 0.95
    entropy_coef: float = 0.01
    clip_coef: float = 0.1
    replay_buffer_size: int = 4096
    batch_size: int = 256
    rollout_length: int = 128
    learning_starts: int = 1000


@dataclasses.dataclass
class Config:
    distributed: Distributed = dataclasses.field(default_factory=Distributed)
    evaluation: Evaluation = dataclasses.field(default_factory=Evaluation)
    training: Training = dataclasses.field(default_factory=Training)
    task: Task = dataclasses.field(default_factory=Task)
    optimizer: Optimizer = dataclasses.field(default_factory=Adam)
    agent: Agent = dataclasses.field(default_factory=PPO)


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)
