import sys

import torch

DEBUG = sys.gettrace() is not None
# env_id = "Pendulum-v1"
# "LunarLanderContinuous-v2"
# MountainCarContinuous-v0
env_id = "inverted_pendulum"
# env_id = "LunarLanderContinuous-v2"
should_render = False
deploy = True
proc_num = 1
host = ""
sweep_yaml = ""
if deploy and not DEBUG:
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"

max_steps = int(1e6)
actor_lr = 5e-3
critic_lr = 5e-3
regularizer = 0.001  # kl regularizer
trajectory_len = 20
num_actors = 1 if DEBUG else 5
batch_size = 32
gamma = 0.99
save_every = 100
grad_clip = 40.

seed = 33
h_dim = 32

device = torch.device("cuda")
