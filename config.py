import sys

import torch

DEBUG = sys.gettrace() is not None
env_id = "CartPole-v0"
should_render = False
deploy = False
proc_num = 1
host = ""
sweep_yaml = ""
if deploy and not DEBUG:
    proc_num = 5
    host = "mila"
    sweep_yaml = "sweep.yaml"

max_steps = int(1e6)
actor_lr = 8e-4
critic_lr = 1e-2
trajectory_len = 5
num_actors = 1 if DEBUG else 5
batch_size = 32
actor_epochs = 5
gamma = 0.99
save_every = 100
grad_clip = 40.
pi_epochs = 5

seed = 33
h_dim = 64

device = torch.device("cpu")
