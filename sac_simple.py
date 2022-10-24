# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import dataclasses
import random
import time

# import gym
import numpy as np
import rlmeta.utils.nested_utils
# import pybullet_envs  # noqa
import torch
import torch.optim as optim
import tqdm
from rlmeta.core.replay_buffer import ReplayBuffer
from rlmeta.samplers import UniformSampler
from rlmeta.storage.circular_buffer import CircularBuffer

import envs
import losses
import models
import wandb


@dataclasses.dataclass
class cfg:
    env_id = "HalfCheetah-v4"
    gamma = 0.99
    tau = 0.005
    batch_size = 256
    learning_starts = 5e3
    replay_buffer_size = int(1e6)
    actor_lr = 3e-4
    critic_lr = 1e-3
    update_epochs = 2
    target_network_frequency = 1
    alpha = 0.2
    train_seed = 1

    # TD3
    exploration_noise = 0.1
    noise_clip = 0.5


# @hydra.main(config_path="conf", config_name="conf_sac")
def main():
    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.train_seed)
    np.random.seed(cfg.train_seed)
    torch.manual_seed(cfg.train_seed)
    torch.cuda.manual_seed(cfg.train_seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    writer = wandb.init(project="impala",
                        mode="disabled"
                        )

    # TODO: this note is wrong. you don't want to compensate for the delay
    # Remove this number of epochs
    # use nsteps

    # env setup
    env = envs.make_control(cfg.env_id, num_envs=1, seed=cfg.train_seed)
    # env = gym.wrappers.RecordEpisodeStatistics(e)

    model = models.SoftActorCritic(env.observation_space.shape, env.action_space.shape, alpha=cfg.alpha).to(device)
    critic_optimizer = optim.Adam(
        [
            {"params": model.critic.parameters()},
            # {"params": model.log_alpha},
        ],
        lr=cfg.critic_lr)
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=cfg.actor_lr)

    # # Automatic entropy tuning
    sampler = UniformSampler()
    sampler.reset(cfg.train_seed)
    rb = ReplayBuffer(CircularBuffer(cfg.replay_buffer_size, collate_fn=torch.cat), sampler)
    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    env_ret = 0
    env_len = 0
    for global_step in tqdm.trange(int(1e6)):
        # Check if you need to sample from the environment

        with torch.no_grad():
            mu, logstd = model.actor(obs.to(device))
            actions, _ = models.to_action(mu, logstd)
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = env.step(actions)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # for info in infos:
        env_ret += rewards
        env_len += 1
        if dones:
            writer.log(
                {
                    "charts/ep_ret": env_ret,
                    "charts/ep_len": env_len,
                }
            )
            env_ret = 0
            env_len = 0
            # env.step(torch.zeros_like(actions))
            dones = dones.logical_not()

        # rb.extend(rlmeta.utils.nested_utils.unbatch_nested(lambda x: x, (obs, actions, rewards, next_obs, dones), 1))
        rb.append([obs, actions, rewards, next_obs, dones])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs.clone()

        if global_step > cfg.learning_starts:
            # ALGO LOGIC: training.
            _, batch, _ = rb.sample(cfg.batch_size)
            batch = rlmeta.utils.nested_utils.map_nested(lambda x: x.to(device), batch)
            qf_loss, critic_info = losses.critic_loss(model, batch)

            critic_optimizer.zero_grad()
            qf_loss.backward()
            critic_optimizer.step()

            if global_step % cfg.update_epochs == 0:  # TD 3 Delayed update support
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                for _ in range(cfg.update_epochs):
                    actor_loss, _ = losses.actor_loss(model, batch)
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    # alpha_loss, _ = losses.alpha_loss(model, batch)
                    # critic_optimizer.zero_grad()
                    # alpha_loss.backward()
                    # critic_optimizer.step()

            # update the target networks
            for (target_key, target_param), (k, param) in zip(model.critic_target.named_parameters(),
                                                              model.critic.named_parameters()):
                assert target_key == k
                target_param.mul_(1.0 - cfg.tau)
                target_param.add_(cfg.tau * param)

            if global_step % 100 == 0:
                SPS = int(global_step / (time.time() - start_time))
                writer.log(
                    {
                        "losses/qf_loss": qf_loss,
                        **critic_info,
                        "losses/actor_loss": actor_loss,
                        "charts/alpha": model.log_alpha.exp(),
                        "charts/SPS": SPS,
                    }
                )

                # writer.log({
                #    "losses/actor_loss": actor_loss.item(),
                #    "losses/alpha_loss": alpha_loss.item(),
                #    "losses/alpha": model.alpha,
                #    **critic_info,
                # }, step=global_step)
                # writer.log({"charts/SPS": int(global_step / (time.time() - start_time))}, step=global_step)

    env.close()
    writer.close()


if __name__ == '__main__':
    main()
