"""Unit tests for src/neuralnet.py (DGM / GPI neural-network HJB solvers).

Uses tiny, locally-defined settings (not the production cfg.NN_*_RUN values,
which are too slow for a test suite) to exercise every code path quickly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.neuralnet import (
    ValueNet,
    _centers_scales,
    _sample_collocation,
    hjb_residual,
    reservation_prices_nn,
    simulate_policy_mc,
    state_bounds,
    train_dgm,
    train_gpi,
)

NU_BAR = 5.0
TINY_PARAMS = dict(
    sigma=0.2, r=0.05, mu=0.05, kappa_I=1.0, eta=0.5, rho=0.0, kappa_J=1.0,
    lambda_bar_P=0.025, lambda_bar_T=0.05, k_A=0.5, k_B=0.5, psi=1.0,
    gamma=0.05, gamma_q=1.0, ell_1=0.0, Q_bar=3.0, nu_bar=NU_BAR, K=100.0,
    T=1.0, n_claims=1.0, epsilon_exec=1e-6, lambda_I=0.0,
)
TINY_N_RATES = 5
RATES = np.linspace(-NU_BAR, NU_BAR, TINY_N_RATES)


def _tiny_net(h_kind: str = "geometric") -> ValueNet:
    centers, scales = _centers_scales(h_kind, TINY_PARAMS["T"])
    return ValueNet(centers, scales)


@pytest.mark.parametrize("h_kind", ["geometric", "arithmetic"])
def test_valuenet_forward_shape_and_finite(h_kind):
    net = _tiny_net(h_kind)
    rng = np.random.default_rng(0)
    batch = _sample_collocation(h_kind, 16, rng, TINY_PARAMS["T"])
    out = net(batch["t"], batch["s"], batch["i"], batch["q"], batch["j"], batch["a"])
    assert out.shape == (16, 1)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("h_kind", ["geometric", "arithmetic"])
def test_hjb_residual_finite_on_fresh_net(h_kind):
    net = _tiny_net(h_kind)
    rng = np.random.default_rng(1)
    batch = _sample_collocation(h_kind, 16, rng, TINY_PARAMS["T"])
    residual = hjb_residual(
        net, 0, h_kind, batch["t"], batch["s"], batch["i"], batch["q"], batch["j"], batch["a"],
        RATES, TINY_PARAMS,
    )
    assert residual.shape == (16, 1)
    assert torch.isfinite(residual).all()


def test_state_bounds_returns_ordered_ranges():
    bounds = state_bounds("geometric")
    for key in ("s", "i", "q", "j", "a"):
        lo, hi = bounds[key]
        assert lo < hi


def test_train_dgm_runs_and_returns_finite_net():
    net = train_dgm(theta=0, h_kind="geometric", params=TINY_PARAMS, n_steps=3, seed=42)
    assert isinstance(net, ValueNet)
    rng = np.random.default_rng(2)
    batch = _sample_collocation("geometric", 8, rng, TINY_PARAMS["T"])
    with torch.no_grad():
        out = net(batch["t"], batch["s"], batch["i"], batch["q"], batch["j"], batch["a"])
    assert torch.isfinite(out).all()


def test_train_gpi_runs_and_policy_in_bounds():
    net, policy = train_gpi(
        theta=0, h_kind="arithmetic", params=TINY_PARAMS, n_outer=2, n_steps_per_iter=3, seed=7,
    )
    assert isinstance(net, ValueNet)
    nu = policy(0.5, 100.0, 0.0, 1.0, 0.0)
    assert isinstance(nu, float)
    assert -NU_BAR <= nu <= NU_BAR


def test_reservation_prices_nn_all_finite():
    nets = {
        theta: train_dgm(theta=theta, h_kind="geometric", params=TINY_PARAMS, n_steps=3, seed=100 + theta)
        for theta in (-1, 0, 1)
    }
    bid, ask, mid, spread = reservation_prices_nn(
        nets, S0=100.0, I0=0.0, Q0=1.0, J0=0.0, r=0.05, t=0.0, T=1.0,
    )
    for value in (bid, ask, mid, spread):
        assert math.isfinite(value)


def test_simulate_policy_mc_runs_and_returns_expected_keys():
    nets_and_policies = {
        theta: train_gpi(theta=theta, h_kind="geometric", params=TINY_PARAMS, n_outer=1, n_steps_per_iter=2, seed=5 + theta)
        for theta in (-1, 0, 1)
    }
    policy_by_theta = {theta: np_pair[1] for theta, np_pair in nets_and_policies.items()}
    result = simulate_policy_mc(
        policy_by_theta, TINY_PARAMS, "geometric", Q0=1.0, J0=0.0,
        n_paths=200, n_steps=5, seed=11,
    )
    assert set(result.keys()) == {"bid", "ask", "mid", "spread"}
    for key, (mean, se) in result.items():
        assert math.isfinite(mean), f"{key} mean not finite"
        assert math.isfinite(se), f"{key} se not finite"
