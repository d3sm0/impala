import dataclasses

from hydra.core.config_store import ConfigStore


@dataclasses.dataclass
class Distributed:
    m_server_name: str = "m_server"
    m_server_addr: str = "127.0.0.1:4411"

    r_server_name: str = "r_server"
    r_server_addr: str = "127.0.0.1:4412"

    c_server_name: str = "c_server"
    c_server_addr: str = "127.0.0.1:4413"

    train_device: str = "cuda:0"
    infer_device: str = "cpu"


@dataclasses.dataclass
class Evaluation:
    num_workers: int = 4
    num_rollouts: int = 8
    seed: int = 456
    deterministic_policy: bool = True


@dataclasses.dataclass
class Training:
    num_workers: int = 8
    num_rollouts: int = 16
    train_seed: int = 123


@dataclasses.dataclass
class DistributedTraining:
    distributed: Distributed = Distributed()
    training: Training = Training()
    evaluation: Evaluation = Evaluation()


@dataclasses.dataclass
class EnvConfig:
    env_library: str = "gym"
    env_name: str = "CartPole-v1"


@dataclasses.dataclass
class Impala:
    actor_lr = 8e-4
    critic_lr = 1e-2
    trajectory_len = 256
    num_actors = 1  # if DEBUG else 5
    batch_size = 64
    actor_epochs = 5
    gamma = 0.99
    lambda_ = 0.95
    save_every = 50
    grad_clip = 40.
    pi_epochs = 5
    seed = 33
    h_dim = 64


@dataclasses.dataclass
class Train:
    max_steps = int(1e6)


@dataclasses.dataclass
class A2CConfig:
    num_envs = 1
    address: str = "127.0.0.1:4532"
    env_id: str = "CartPole-v1"
    total_steps: int = int(1e6)
    batch_size: int = 8
    rollout_length: int = 5
    lr: float = 7e-4
    rmsprop_eps: float = 1e-5
    alpha: float = 0.99
    gamma: float = 0.99
    grad_clip: float = 0.5
    entropy_cost: float = 0.
    seed: int = 33
    evaluate_every: int = 100


@dataclasses.dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = int(1e6)
    num_envs: int = 1
    batch_size: int = 32
    rollout_length: int = 2048
    epochs: int = 10
    clip_coef: float = 0.2
    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    grad_clip: float = 1.
    entropy_cost: float = 0.
    kl_cost: float = 0.01
    seed: int = 33
    evaluate_every: int = 100


@dataclasses.dataclass
class PPOVtrace:
    env_id: str = "Pong-v5"
    total_steps: int = int(1e6)
    num_envs: int = 8
    batch_size: int = 128
    rollout_length: int = batch_size
    epochs: int = 4
    lr: float = 2.5e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    lambda_: float = 1.
    grad_clip: float = 0.5
    entropy_cost: float = 0.01
    kl_cost: float = 0.01
    seed: int = 33
    evaluate_every: int = 100


@dataclasses.dataclass
class PPEConfig:
    env_id: str = "starpilot"
    lr: float = 5e-4
    total_steps: int = int(25e6)
    num_envs: int = 16
    rollout_length: int = 256
    gamma: float = 0.999
    lambda_: float = 0.95
    minibatch_size: int = 8
    batch_size: int = (num_envs * rollout_length) // minibatch_size
    epochs: int = 1
    entropy_cost: float = 0.01
    kl_cost: float = 0.01
    grad_clip: float = 0.5
    update_every: int = 32
    aux_buffer_size: int = num_envs * update_every
    aux_epochs: int = 6
    seed: int = 33
    adam_eps: float = 1e-5
    evaluate_every: int = 100


cs = ConfigStore.instance()
cs.store(name="a2c_config", node=A2CConfig)
cs.store(name="ppo_config", node=PPOConfig)
cs.store(name="ppe_config", node=PPEConfig)
cs.store(name="ppovtrace_config", node=PPOVtrace)
