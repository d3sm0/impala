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
import models
import utils

logger = utils.get_logger("main")


def evaluate_loss(model, batch):
    policy = model.actor(batch.state)
    v_tm1 = model.critic(batch.state)
    v_t = model.critic(batch.next_state)
    r_t = batch.reward
    not_done = (1 - batch.done)
    pi_old = torch_dist.Categorical(logits=batch.info)  # TODO fix this
    rho_tm1 = (policy.log_prob(batch.action) - pi_old.log_prob(batch.action)).exp().detach()
    adv, errors, _ = vmap(rlego.vtrace_td_error_and_advantage)(v_tm1, v_t, r_t, not_done * config.gamma, rho_tm1)

    pi_grad = (- policy.log_prob(batch.action) * adv).sum(1).mean(0)
    td = 0.5 * (errors.pow(2)).sum(1).mean(0)
    kl = torch_dist.kl_divergence(policy, pi_old).sum(1).mean(0)
    loss = pi_grad + 0.5 * td + 0.01 * kl
    entropy = policy.entropy().sum(0).mean()
    return loss, tree.map_structure(lambda x: x.detach().numpy(),
                                    {"pi_loss": pi_grad, "td": td, "rho": rho_tm1.mean(), "kl": kl, "entropy": entropy})


def run_learner(model_queue, data_queue, writer_queue, frame_counter):
    env = utils.GymWrapper(gym.make(config.env_id))
    env.seed(config.seed)
    model = models.Actor(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, h_dim=config.h_dim)
    state, *_ = env.reset()
    transition = (state, torch.tensor(0.), torch.tensor(0.), torch.zeros_like(state), torch.tensor(0.),
                  torch.zeros(env.action_space.n))
    trajectory = collections.deque(maxlen=config.trajectory_len + 1)
    for step in itertools.count():
        trajectory_len = config.trajectory_len + int(step == 0)
        if not model_queue.empty():
            state_dict = model_queue.get()
            if state_dict is None:
                break
            else:
                logger.info(f"Policy update at {frame_counter.value}")
                model.load_state_dict(state_dict)
        trajectory.append(transition)
        _sample_trajectory(env, model, trajectory, writer_queue, trajectory_len)
        with threading.Lock():
            frame_counter.value = frame_counter.value + trajectory_len
        data_queue.put(tree.map_structure(lambda *x: torch.stack(x), *trajectory))


@torch.no_grad()
def _sample_trajectory(env, model, trajectory, writer_queue, trajectory_len):
    state = trajectory[-1][0]
    for t in range(trajectory_len):
        pi = model(state)
        action = pi.sample()
        next_state, reward, done, info = env.step(action.numpy())
        transition = (state, action, reward, next_state, done, pi.logits)
        trajectory.append(transition)
        state = next_state
        if "step" in info.keys():
            writer_queue.put(info)


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
    # batch = specs.Transition(*tree.map_structure(lambda *x: torch.stack(x), *batch))
    return batch


def writer_loop(writer, writer_queue, frame_counter):
    while True:
        infos = writer_queue.get()
        if infos is None:
            break
        writer.add_scalars(infos, global_step=frame_counter.value)


def train(model, optimizer, writer):
    data_queue = multiprocessing.SimpleQueue()
    model_queue = multiprocessing.SimpleQueue()
    writer_queue = multiprocessing.SimpleQueue()
    frame_counter = multiprocessing.Value('i', 0)
    model_queue.put(model.actor.state_dict())
    actor_procs = [
        multiprocessing.Process(target=run_learner, args=(model_queue, data_queue, writer_queue, frame_counter)) for _
        in range(config.num_actors)]
    for p in actor_procs:
        p.start()

    writer_thread = threading.Thread(target=writer_loop, args=(writer, writer_queue, frame_counter))
    writer_thread.start()
    try:
        train_loop(data_queue, model, model_queue, optimizer, writer_queue)
    except KeyboardInterrupt:
        print("close")
    finally:
        writer_queue.put(None)
        for p in actor_procs:
            p.terminate()
        logger.error("Waiting for the test workers to finish.")
        for p in actor_procs:
            p.join()


def train_loop(data_queue, model, model_queue, optimizer, writer_queue):
    buffer = rlego.Buffer(config.max_steps)
    for global_step in itertools.count():
        with utils.timer() as t:
            batch = collect_transitions(data_queue, config.batch_size)
        buffer.extend(batch)  # noqa
        logger.info(f"{global_step} in dt:{t():.2f}")
        loss, loss_info = evaluate_loss(model, buffer.sample(batch_size=config.batch_size))  # noqa
        opt_info = update_params(optimizer, model, loss)
        model_queue.put(model.actor.state_dict())
        writer_queue.put({**opt_info, **loss_info})


def main():
    buddy.register_defaults(config.__dict__)
    writer = buddy.deploy(
        proc_num=config.proc_num,
        host=config.host,
        sweep_yaml=config.sweep_yaml,
        disabled=config.DEBUG,
        wandb_kwargs={"project": "impala"},
        extra_modules=["python/3.7", "cuda/11.1/cudnn/8.0"],
    )

    env = gym.make(config.env_id)
    model = models.Agent(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, h_dim=config.h_dim)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.policy_lr, momentum=0., eps=0.01)
    del env

    train(model, optimizer, writer)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()
