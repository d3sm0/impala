import sys

import torch

DEBUG = sys.gettrace() is not None
env_id = "CartPole-v0"
should_render = False
if DEBUG:
    proc_num = 1
    host = ""
    sweep_yaml = ""
else:
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"

max_steps = int(1e6)
actor_lr = 0.0001
critic_lr = 0.001
trajectory_len = 20
num_actors = 5
batch_size = 32
gamma = 0.99
save_every = 100
grad_clip = 40.

seed = 33
h_dim = 64

device = torch.device("cpu")
