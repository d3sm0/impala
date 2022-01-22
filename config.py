import sys

import torch

DEBUG = sys.gettrace() is not None
env_id = "CartPole-v0"
should_render = False
proc_num = 1
host = ""
sweep_yaml = ""  # k"sweep.yaml"

max_steps = int(1e6)
actor_lr = 0.001
critic_lr = 0.005
trajectory_len = 21
num_actors = 1 if DEBUG else 5
batch_size = 32
gamma = 0.99
save_every = 100
grad_clip = 40.

seed = 33
h_dim = 64

device = torch.device("cpu")
