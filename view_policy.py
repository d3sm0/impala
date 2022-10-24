import os

import gym
import torch
import wandb

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-11.7/lib64:/usr/lib/nvidia:/home/d3sm0/.mujoco/mujoco210/bin"


def main(run_id="hyjfuvgq"):
    api = wandb.Api()
    wandb_run = api.run(f"d3sm0/impala/{run_id}")
    src_dir = os.path.join("runs", run_id)
    os.makedirs(src_dir, exist_ok=True)
    fname = wandb_run.file("model-100.pt").download(replace=True, root=src_dir).name

    actor, critic = torch.load(fname, map_location=torch.device('cpu'))
    env = gym.make("HalfCheetah-v4", render_mode="rgb_array_list", disable_env_checker=True)
    env = gym.wrappers.RecordVideo(env, os.path.join(src_dir, "videos"), name_prefix=wandb_run.name)
    obs, _ = env.reset(seed=123)
    with torch.no_grad():
        while True:
            action, _ = actor(torch.tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, _ = env.step(action.numpy())
            if terminated or truncated:
                break
        env.close()
    wandb_run.upload_file(os.path.join(env.video_folder, env.name_prefix + "-episode-0.mp4"))


if __name__ == '__main__':
    main( run_id="w00o7ozi")
