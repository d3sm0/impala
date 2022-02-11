import collections
import itertools
import multiprocessing
import threading

import experiment_buddy as buddy
import gym
import rlego
import torch
import torch.distributions as torch_dist
import tree
from torch._vmap_internals import vmap

import config
import lqr
import models
import specs
import utils

logger = utils.get_logger("main")

torch.autograd.set_detect_anomaly(True)


def evaluate_critic_loss(model, batch):
    policy = model.actor(batch.state)
    v_tm1 = model.critic(batch.state)
    q_tm1 = model.q(batch.state, batch.action)
    with torch.no_grad():
        v_t = model.critic(batch.next_state)
    r_t = batch.reward
    not_done = (1 - batch.done)
    mask = torch.roll(not_done, 1, dims=(0,))
    pi_old = torch_dist.Normal(*torch.split(batch.logits, split_size_or_sections=1, dim=-1))
    rho_tm1 = (policy.log_prob(batch.action) - pi_old.log_prob(batch.action)).sum(dim=-1).exp()
    adv, v_target, q_target = vmap(rlego.vtrace_td_error_and_advantage)(v_tm1.detach(), v_t, r_t,
                                                                        not_done * config.gamma, rho_tm1.detach())

    td = 0.5 * (mask * (v_target * (1 - config.gamma) - v_tm1).pow(2)).sum(1).mean()
    q_loss = 0.5 * (mask * (q_target * (1 - config.gamma) - q_tm1).pow(2)).sum(1).mean()
    return (td + q_loss), tree.map_structure(lambda x: x.detach().numpy(), {
        "td": td,
        "rho": rho_tm1.mean(),
        "q_loss": q_loss
    })


def evaluate_actor_loss(model, batch):
    pi_old = torch_dist.Normal(*torch.split(batch.logits, split_size_or_sections=1, dim=-1))
    not_done = (1 - batch.done)
    mask = torch.roll(not_done, 1, dims=(0,))
    policy = model.actor(batch.state)
    pi_grad = (-model.q(batch.state, policy.rsample())).sum(1).mean()
    kl = (torch_dist.kl_divergence(policy, pi_old).sum(dim=-1) * mask).sum(1).mean()
    entropy = (policy.entropy().sum(dim=-1) * mask).mean()
    w = w_gaussian(policy, pi_old).sum(1).mean()
    return pi_grad, tree.map_structure(lambda x: x.detach().numpy(),
                                       {"pi_loss": pi_grad,
                                        "kl": kl,
                                        "entropy": entropy,
                                        "scale": policy.scale.mean(), "loc": policy.mean.mean(),
                                        "wasserstein": w,
                                        })


def w_gaussian(pi, pi_k):
    loss = (pi.loc - pi_k.loc).pow(2) - (pi.variance + pi_k.variance - 2 * (pi.variance * pi_k.variance).sqrt())
    return loss.clamp_min(0).sum(dim=-1)


def run_learner(model_queue, data_queue, writer_queue, frame_counter, proc_id):
    env = gym.make(config.env_id)
    # env = lqr.Lqg()
    env = utils.GymWrapper(env)
    env.unwrapped.seed(config.seed)
    utils.set_seed(config.seed)
    model = models.Actor(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                         h_dim=config.h_dim)
    state, *_ = env.reset()
    trajectory = collections.deque(maxlen=config.trajectory_len + 1)
    for step in itertools.count():
        trajectory_len = config.trajectory_len + int(step == 0)
        if not model_queue.empty():
            state_dict = model_queue.get()
            if state_dict is None:
                break
            else:
                with utils.timer() as t:
                    model.actor.load_state_dict(state_dict)
                logger.info(
                    f"Update Policy {proc_id}. F:{frame_counter.value}, steps:{frame_counter.value}, dt {t():.2f}.")
        state = _sample_trajectory(state, env, model, trajectory, writer_queue, trajectory_len, proc_id)
        with threading.Lock():
            frame_counter.value = frame_counter.value + trajectory_len
        data_queue.put(tree.map_structure(lambda *x: torch.stack(x).share_memory_(), *trajectory))


@torch.no_grad()
def _sample_trajectory(state, env, model, trajectory, writer_queue, trajectory_len, proc_id):
    for t in range(trajectory_len):
        pi = model.actor(state)
        action = pi.sample()
        next_state, reward, done, info = env.step(action)
        transition = (state, action, reward, next_state, done, torch.cat([pi.loc, pi.scale]))
        trajectory.append(transition)
        state = next_state
        if "step" in info.keys():
            writer_queue.put((proc_id, info))
    return state


def update_params(optimizer, model, loss):
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return {"grad_norm": grad_norm}


def collect_transitions(data_queue, batch_size):
    batch = []
    for b in range(batch_size):
        transitions = data_queue.get()
        batch.append(transitions)
    batch = specs.Transition(*tree.map_structure(lambda *x: torch.stack(x), *batch))
    return batch


def writer_loop(writer, writer_queue, frame_counter):
    while True:
        infos = writer_queue.get()
        if infos is None:
            break
        proc_id, infos = infos
        if proc_id == 0:
            writer.add_scalars(infos, global_step=frame_counter.value)


def train(model, critic_optimizer, actor_optimizer, writer):
    data_queue = torch.multiprocessing.SimpleQueue()
    writer_queue = multiprocessing.SimpleQueue()
    frame_counter = multiprocessing.Value('i', 0)
    model_queues = {}
    actor_procs = []
    for proc_idx in range(config.num_actors):
        model_queue = torch.multiprocessing.SimpleQueue()
        model_queue.put(model.actor.state_dict())
        p = multiprocessing.Process(target=run_learner,
                                    args=(model_queue, data_queue, writer_queue, frame_counter, proc_idx))
        model_queues[proc_idx] = model_queue
        actor_procs.append(p)
    for p in actor_procs:
        p.start()

    writer_thread = threading.Thread(target=writer_loop, args=(writer, writer_queue, frame_counter))
    writer_thread.start()
    try:
        train_loop(data_queue, model, model_queues, critic_optimizer, actor_optimizer, writer_queue, frame_counter)
    except KeyboardInterrupt:
        print("close")
    finally:
        writer_queue.put(None)
        for p in actor_procs:
            p.terminate()
        logger.error("Waiting for the test workers to finish.")
        for p in actor_procs:
            p.join()


def train_loop(data_queue, model, model_queues, critic_optimizer, actor_optimizer, writer_queue, frame_counter):
    for global_step in itertools.count():
        with utils.timer() as t:
            batch = collect_transitions(data_queue, config.batch_size)
        loss, loss_info = evaluate_loss(model, batch)  # noqa
        if global_step % 100 == 0:
            logger.info(f"Frames: {frame_counter.value}. step: {global_step}. dt:{t():.2f}")
        opt_info = update_params(optimizer, model, loss)
        logger.info(f"{global_step} in dt:{t():.2f}")
        value_loss, critic_info = evaluate_critic_loss(model, batch)  # noqa
        critic_opt_info = update_params(critic_optimizer, model, value_loss)
        actor_loss, actor_info = evaluate_actor_loss(model, batch)  # noqa
        actor_opt_info = update_params(actor_optimizer, model, actor_loss)
        for proc_idx, model_queue in model_queues.items():
            model_queue.put(model.actor.state_dict())
        writer_queue.put((0, {**critic_opt_info, **actor_opt_info, **critic_info, **actor_info}))
        if frame_counter.value >= config.max_steps:
            break


def main():
    buddy.register_defaults(config.__dict__)
    writer = buddy.deploy(
        proc_num=config.proc_num,
        host=config.host,
        sweep_definition=config.sweep_yaml,
        disabled=config.DEBUG,
        wandb_kwargs={"project": "impala"},
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = gym.make(config.env_id)
    # env = lqr.Lqg()
    model = models.Agent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                         h_dim=config.h_dim)

    actor_optimizer = torch.optim.Adam([{"params": model.actor.parameters(), "lr": config.actor_lr}])
    critic_optimizer = torch.optim.Adam([{"params": model.critic.parameters(), "lr": config.critic_lr},
                                         {"params": model.q.parameters(), "lr": config.critic_lr}
                                         ])
    del env

    train(model, critic_optimizer, actor_optimizer, writer)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    main()
