# import functorch
import functorch
import rlego
import torch

import models.distributed_models


# TODO: refactor losses

# batched_vtrace = functorch.vmap(rlego.vtrace_td_error_and_advantage)
# batched_gae = functorch.vmap(rlego.truncated_generalized_advantage_estimation)
# batched_is = functorch.vmap(rlego.importance_corrected_td_errors)


def get_muesli(q, v, pi, n_samples=16):
    actions = models.distributed_models.sample(sample_shape=(n_samples,))
    one_hot = torch.nn.functional.one_hot(actions, num_classes=q.shape[-1]).float()
    q_values = (one_hot * q).sum(-1)
    adv = torch.clamp(q_values - v, -1, 1).exp()
    num = adv * pi.log_prob(actions)
    denom = 1 / n_samples * (1 + adv.sum(dim=1, keepdims=True))
    return - (num / denom).mean(0)


def muesli_loss(batch, model, gamma=0.99, lambda_=0.95):
    s, a, r, s1, d, pi_ref = batch
    mask = (1 - d.to(torch.float32)).roll(shifts=1, dims=1)
    discount_t = (1 - d.to(torch.float32)) * gamma

    pi_tm1, v_tm1 = model(s)
    pi_tm1 = torch.distributions.Categorical(logits=pi_tm1)
    pi_ref = torch.distributions.Categorical(logits=pi_ref)

    ratio = torch.exp(pi_tm1.log_prob(a) - pi_ref.log_prob(a))
    with torch.no_grad():
        _, v_boostrap = model(s1[:, -1:])
        v_t = torch.cat([v_tm1[:, 1:], v_boostrap], dim=1)
    adv, err, q_target = batched_vtrace(v_tm1, v_t, r, discount_t, ratio.detach(),
                                        lambda_=lambda_)
    q_tm1 = model.q(s)
    td_loss = 0.5 * (mask * err).pow(2).mean()

    q_selected = (torch.nn.functional.one_hot(a, num_classes=q_tm1.shape[-1]).float() * q_tm1).sum(-1)
    q_loss = 0.5 * (mask * (q_target - q_selected)).pow(2).mean()
    muesli_reg = get_muesli(q_tm1.detach(), v_tm1.detach(), pi_tm1, n_samples=16).mean()

    pg_loss = -(mask * adv * ratio).mean()
    kl = torch.distributions.kl_divergence(pi_tm1, pi_ref).mean().clamp_min(0.)
    entropy = pi_tm1.entropy().mean()
    loss = pg_loss + td_loss + q_loss + muesli_reg
    return loss, {
        "train/muesli": muesli_reg.detach(),
        "train/loss": loss.detach(),
        "train/entropy": entropy.detach(),
        "train/td": td_loss.detach(),
        "train/pg": pg_loss.detach(),
        "train/kl": kl.detach(),
        "train/ratio": ratio.mean().detach(),
    }


def ppo_loss_is(batch, model, entropy_cost=0.01, clip_coeff=0.1, lambda_=0.95, gamma=0.99):
    s, a, r, s1, d, pi_ref = batch
    mask = (1 - d).roll(shifts=1, dims=1)
    discount_t = (1 - d.float()) * gamma
    pi_tm1, v_tm1 = functorch.vmap(model)(s)
    pi_tm1 = torch.distributions.Categorical(logits=pi_tm1)
    pi_ref = torch.distributions.Categorical(logits=pi_ref)
    ratio = torch.exp(pi_tm1.log_prob(a) - pi_ref.log_prob(a))
    with torch.no_grad():
        _, v_boostrap = functorch.vmap(model)(s1[:, -1:])
        v_t = torch.cat([v_tm1[:, 1:], v_boostrap], dim=1)
    adv, err, _ = batched_vtrace(v_tm1.squeeze(-1), v_t.squeeze(-1), r, discount_t, ratio.detach(), lambda_=lambda_)
    td_loss = 0.5 * (mask * err).pow(2).mean()
    adv = (adv * mask).detach()
    pg_loss_1 = -(adv * ratio)
    pg_loss_2 = -torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff) * adv
    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
    kl = torch.distributions.kl_divergence(pi_tm1, pi_ref).mean().clamp_min(0.)
    entropy = pi_tm1.entropy().mean()
    loss = pg_loss + td_loss - entropy_cost * entropy
    return loss, {
        "train/loss": loss.detach(),
        "train/entropy": entropy.detach(),
        "train/td": td_loss.detach(),
        "train/pg": pg_loss.detach(),
        "train/kl": kl.detach(),
        "train/ratio": ratio.mean().detach(),
    }


def impala_loss(batch, model, lambda_=1., entropy_cost=0.01):
    s, a, r, discount_t, pi_ref = batch
    pi_tm1, v_tm1 = model(s)
    # pi_tm1 = torch.distributions.Categorical(logits=pi_tm1)
    # pi_ref = torch.distributions.Categorical(logits=pi_ref)
    # ratio = torch.exp(pi_tm1.log_prob(a.squeeze(dim=-1)) - pi_ref.log_prob(a.squeeze(dim=-1)))
    a = a.squeeze(dim=-1)
    ratio = torch.ones_like(r).squeeze().float()

    # with torch.no_grad():
    #    v_t = model(s[-1:])[1].squeeze(-1)
    #    v_t = torch.cat([v_tm1[1:].squeeze(-1), v_t], dim=0)

    v_t = v_tm1.squeeze(-1).detach()
    adv, err, _ = rlego.vtrace_td_error_and_advantage(v_tm1.squeeze(-1), v_t.squeeze(dim=-1), r.squeeze(dim=-1),
                                                      discount_t.squeeze(dim=-1), ratio.detach(),
                                                      lambda_=lambda_)
    td_loss = 0.5 * err.pow(2).mean()

    # pg_loss_1 = -(adv * ratio)
    # pg_loss_2 = -torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff) * adv
    # pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
    # pg_loss = - (torch.log(torch.clamp(ratio, 1 / (1 + clip_coeff), 1 + clip_coeff)) * adv).mean()
    # pg_loss = -(pi_tm1.log_prob(a.squeeze(dim=-1)) * adv).sum()
    pg_loss = - (pi_tm1.gather(1, a.unsqueeze(dim=-1)).squeeze(dim=-1) * adv).sum()

    # kl = torch.distributions.kl_divergence(pi_tm1, pi_ref).mean().clamp_min(0.)
    # entropy = pi_tm1.entropy().mean()
    loss = pg_loss + td_loss  # - entropy_cost * entropy
    return loss, {
        "train/loss": loss.detach(),
        # "train/entropy": entropy.detach(),
        "train/td": td_loss.detach(),
        "train/pg": pg_loss.detach(),
        # "train/kl": kl.detach(),
        "train/ratio": ratio.mean().detach(),
    }


def ppo_loss(model, batch, entropy_cost=0.01, clip_coeff=0.1):
    s, a, v_target, pi_ref = batch
    pi_tm1, v_tm1 = model(s)
    pi_tm1 = torch.distributions.Categorical(logits=pi_tm1)
    pi_ref = torch.distributions.Categorical(logits=pi_ref)
    ratio = torch.exp(pi_tm1.log_prob(a) - pi_ref.log_prob(a))
    adv = (v_target - v_tm1.squeeze(-1))
    td_loss = 0.5 * adv.pow(2).mean()
    adv = adv.detach()
    pg_loss_1 = -(adv * ratio)
    pg_loss_2 = -torch.clamp(ratio, 1 - clip_coeff, 1 + clip_coeff) * adv
    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
    # pg_loss = - (torch.log(torch.clamp(ratio, 1 / (1 + clip_coeff), 1 + clip_coeff)) * adv).mean()
    kl = torch.distributions.kl_divergence(pi_tm1, pi_ref).mean().clamp_min(0.)
    entropy = pi_tm1.entropy().mean()
    loss = pg_loss + td_loss - entropy_cost * entropy
    return loss, {
        "train_step/loss": loss.detach(),
        "train_step/entropy": entropy.detach(),
        "train_step/td": td_loss.detach(),
        "train_step/pg": pg_loss.detach(),
        "train_step/target": v_target.mean().detach(),
        "train_step/kl": kl.detach(),
        "train_step/ratio": ratio.mean().detach(),
    }
