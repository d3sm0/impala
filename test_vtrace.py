# ported from rlax/_src/vtrace_test.py
import functools

import numpy as np
import rlego
import torch
import torch.distributions as torch_dist
from absl.testing import absltest
from absl.testing import parameterized
from torch._vmap_internals import vmap


def categorical_importance_sampling_ratios(target_policy, behavior_policy, actions):
    pi = torch_dist.Categorical(logits=target_policy).log_prob(actions)
    pi_old = torch_dist.Categorical(logits=behavior_policy).log_prob(actions)
    return (pi - pi_old).exp()


class VTraceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        behavior_policy_logits = torch.tensor(
            [[[8.9, 0.7], [5.0, 1.0], [0.6, 0.1], [-0.9, -0.1]],
             [[0.3, -5.0], [1.0, -8.0], [0.3, 1.7], [4.7, 3.3]]],
            dtype=torch.float32)
        target_policy_logits = torch.tensor(
            [[[0.4, 0.5], [9.2, 8.8], [0.7, 4.4], [7.9, 1.4]],
             [[1.0, 0.9], [1.0, -1.0], [-4.3, 8.7], [0.8, 0.3]]],
            dtype=torch.float32)
        actions = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=torch.int32)
        self._rho_tm1 = categorical_importance_sampling_ratios(
            target_policy_logits, behavior_policy_logits, actions)
        self._rewards = torch.tensor(
            [[-1.3, -1.3, 2.3, 42.0],
             [1.3, 5.3, -3.3, -5.0]],
            dtype=torch.float32)
        self._discounts = torch.tensor(
            [[0., 0.89, 0.85, 0.99],
             [0.88, 1., 0.83, 0.95]],
            dtype=torch.float32)
        self._values = torch.tensor(
            [[2.1, 1.1, -3.1, 0.0],
             [3.1, 0.1, -1.1, 7.4]],
            dtype=torch.float32)
        self._bootstrap_value = torch.tensor([8.4, -1.2], dtype=torch.float32)
        self._inputs = [
            self._rewards, self._discounts, self._rho_tm1,
            self._values, self._bootstrap_value]

        self._clip_rho_threshold = 1.0
        self._clip_pg_rho_threshold = 5.0
        self._lambda = 1.0

        self._expected_td = torch.tensor(
            [[-1.6155143, -3.4973226, 1.8670533, 5.0316002e1],
             [1.4662437, 3.6116405, -8.3327293e-5, -1.3540000e1]],
            dtype=torch.float32)
        self._expected_pg = torch.tensor(
            [[-1.6155143, -3.4973226, 1.8670534, 5.0316002e1],
             [1.4662433, 3.6116405, -8.3369283e-05, -1.3540000e+1]],
            dtype=torch.float32)

    def test_vtrace_td_error_and_advantage(self):
        """Tests for a full batch."""
        vtrace_td_error_and_advantage = vmap(functools.partial(rlego.vtrace_td_error_and_advantage,
                                                               clip_rho_threshold=self._clip_rho_threshold,
                                                               lambda_=self._lambda))
        # Get function arguments.
        r_t, discount_t, rho_tm1, v_tm1, bootstrap_value = self._inputs
        v_t = torch.cat([v_tm1[:, 1:], bootstrap_value[:, None]], dim=1)
        # Compute vtrace output.
        (pg_advantage, errors, _) = vtrace_td_error_and_advantage(
            v_tm1, v_t, r_t, discount_t, rho_tm1)
        # Test output.
        np.testing.assert_allclose(
            self._expected_td, errors, rtol=1e-3)
        np.testing.assert_allclose(
            self._expected_pg, pg_advantage, rtol=1e-3)

    def test_lambda_q_estimate(self):
        """Tests for a full batch."""
        lambda_ = 0.8
        vtrace_td_error_and_advantage = vmap(functools.partial(
            rlego.vtrace_td_error_and_advantage,
            clip_rho_threshold=self._clip_rho_threshold, lambda_=lambda_))
        # Get function arguments.
        r_t, discount_t, rho_tm1, v_tm1, bootstrap_value = self._inputs
        v_t = torch.cat([v_tm1[:, 1:], bootstrap_value[:, None]], dim=1)
        # Compute vtrace output.
        (pg_advantage, errors, q_estimate) = vtrace_td_error_and_advantage(
            v_tm1, v_t, r_t, discount_t, rho_tm1)
        expected_vs = errors + v_tm1
        clipped_rho_tm1 = np.minimum(self._clip_rho_threshold, rho_tm1)
        vs_from_q = v_tm1 + clipped_rho_tm1 * (q_estimate - v_tm1)
        # Test output.
        np.testing.assert_allclose(expected_vs, vs_from_q, rtol=1e-3)


if __name__ == '__main__':
    absltest.main()
