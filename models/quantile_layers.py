import torch
import torch.nn as nn

import models.common


class ImplicitQuantileHead(nn.Module):
    """See arXiv: 1806.06923
    This module, in combination with quantile loss,
    learns how to generate the quantile of the distribution of the target.
    A quantile value, tau, is randomly generated with a Uniform([0, 1])).
    This quantile value is embedded in this module and also passed to the quantile loss:
    this should force the model to learn the appropriate quantile.
    """

    def __init__(self, input_dim, action_dim, d_model):
        super(ImplicitQuantileHead, self).__init__()
        self.quantile_layer = QuantileLayer(input_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            models.common.layer_init_truncated(nn.Linear(input_dim, d_model)),
            nn.GELU()
        )
        self.q = models.common.layer_init_truncated(nn.Linear(d_model, action_dim))
        self.v = models.common.layer_init_truncated(nn.Linear(d_model, 1))

    def forward(self, input_data: torch.Tensor, tau: torch.Tensor = torch.tensor((0.5,))):
        tau = tau.to(input_data.device)
        embedded_tau = self.quantile_layer(tau)
        # This can also be written as a multiplication
        h = input_data.unsqueeze(1) * embedded_tau
        h = self.project(h)
        q = self.q(h)
        v = self.v(h)
        return v + q - q.mean(-1, keepdim=True)


class QuantileLayer(nn.Module):
    """Define quantile embedding layer, i.e. phi in the IQN paper (arXiv: 1806.06923)."""

    def __init__(self, num_output):
        super(QuantileLayer, self).__init__()
        self.n_cos_embedding = 64
        self.num_output = num_output
        self.output_layer = nn.Sequential(
            models.common.layer_init_truncated(nn.Linear(self.n_cos_embedding, num_output)),
            nn.GELU(),
        )
        self.register_buffer("embedding_range",
                             torch.arange(1, self.n_cos_embedding + 1).reshape(1, 1, self.n_cos_embedding))

    def forward(self, tau):
        cos_embedded_tau = torch.cos(torch.pi * self.embedding_range * tau.unsqueeze(-1))
        final_output = self.output_layer(cos_embedded_tau)
        return final_output


def cpw(tau, eta=0.71):
    return tau ** eta / (tau ** eta + (1 - tau) ** eta) ** -eta


def uniform(tau):
    return tau


def cvar(tau, eta=0.1):
    return tau * eta


def pow(tau, eta=-2.):
    p = (1 / 1 + abs(eta))
    if eta > 0:
        return tau ** p
    return (1 - (1 - tau)) ** p


def wang(tau, eta=-0.75):
    # eta negative risk averse
    phi = torch.distributions.Normal(0, 1)
    return phi.cdf(phi.icdf(tau) + eta)


# TODO: import this from rlego
def quantile_regression_loss(dist_src: torch.Tensor, tau_src: torch.Tensor, dist_target: torch.Tensor,
                             huber_param: float = 1.):
    delta = dist_target.unsqueeze(1) - dist_src.unsqueeze(2)
    weight = torch.abs(tau_src.unsqueeze(2) - (delta < 0.).to(torch.float32)).detach()
    if huber_param == 0:
        delta = torch.abs(delta)
    else:
        abs_delta = torch.abs(delta)
        quadratic = torch.clamp_max(abs_delta, huber_param)
        delta = 0.5 * quadratic.pow(2) + huber_param * (abs_delta - quadratic)
    return (weight * delta).sum(1).mean(-1)
