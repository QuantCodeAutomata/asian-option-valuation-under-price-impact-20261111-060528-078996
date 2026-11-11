"""Unit tests for src/bellman.py (discrete dealer Bellman scheme, Sec. 5.1 (83)-(88))."""

import numpy as np
import pytest

from src.bellman import (
    h_func,
    interp5d,
    make_grids,
    phi_func,
    reservation_prices,
    simulate_deterministic_path,
    solve_bellman,
)

# A deliberately tiny grid: fast enough for unit tests, large enough to
# exercise every code path (interpolation corners, admissible-rate clamping).
TINY = dict(n_s=5, n_i=3, n_q=5, n_j=3, n_a=7)
BASE = dict(S0=100.0, I0=0.0, T=1.0, sigma=0.2, eta=0.5, kappa_I=1.0, Q_bar=3.0,
            nu_bar=5.0, kappa_J=1.0, K=100.0)


def _grids(h_kind="geometric"):
    return make_grids(**BASE, h_kind=h_kind, **TINY)


def test_phi_func_payoff_shapes():
    a = np.array([0.0, np.log(100.0), np.log(150.0)])
    arith = phi_func(np.exp(a) - 1.0, T=1.0, K=100.0, h_kind="arithmetic")
    geo = phi_func(a, T=1.0, K=100.0, h_kind="geometric")
    assert np.all(arith >= 0.0) and np.all(geo >= 0.0)
    assert geo[-1] > 0.0  # exp(log(150)) = 150 > K


def test_h_func_arithmetic_vs_geometric():
    s = np.array([80.0, 100.0, 120.0])
    assert np.allclose(h_func(s, "arithmetic"), s)
    assert np.allclose(h_func(s, "geometric"), np.log(s))


def test_interp5d_exact_at_grid_nodes():
    grids = _grids()
    values = np.random.default_rng(0).normal(size=(5, 3, 5, 3, 7))
    q = (grids.s[2:3], grids.i[1:2], grids.q[2:3], grids.j[1:2], grids.a[3:4])
    out = interp5d(grids, values, q)
    assert out[0] == pytest.approx(values[2, 1, 2, 1, 3])


def test_interp5d_clamps_outside_grid():
    grids = _grids()
    values = np.ones((5, 3, 5, 3, 7))
    far = (np.array([grids.s[-1] * 10]), grids.i[:1], grids.q[:1], grids.j[:1], grids.a[:1])
    out = interp5d(grids, values, far)
    assert out[0] == pytest.approx(1.0)


@pytest.mark.parametrize("h_kind", ["geometric", "arithmetic"])
def test_solve_bellman_runs_and_orders_quotes(h_kind):
    """Bid <= ask, and the dealer's buy/sell quotes bracket the no-impact
    (theta=0) value, matching the reservation-price definition (76)."""
    grids = _grids(h_kind)
    rates = np.linspace(-5.0, 5.0, 5)
    out = solve_bellman(
        (-1, 0, 1), grids, N=3, T=1.0, K=100.0, n_claims=1.0, ell_1=0.0, Gamma_Q=1.0,
        rates=rates, r=0.05, mu=0.05, sigma=0.2, lambda_I=0.0, kappa_I=1.0, eta=0.5,
        kappa_J=1.0, rho=0.0, gamma=0.05, lambda_bar_P=0.025, lambda_bar_T=0.05,
        k_A=0.5, k_B=0.5, psi=1.0, epsilon_exec=1e-6, store_policy=True,
    )
    bid, ask, mid, spread = reservation_prices(out, grids, 100.0, 0.0, 0.0, 0.0, 0.05, 0.0, 1.0)
    assert spread >= -1e-8
    assert ask >= bid - 1e-8
    assert np.isfinite(mid)


def test_solve_bellman_spread_increases_with_cost():
    """A higher trading cost coefficient k widens the dealer's spread at Q0=0
    (more costly to unwind/hedge inventory), consistent with Table 4 Panel A."""
    rates = np.linspace(-5.0, 5.0, 5)

    def spread_at(k):
        grids = _grids("geometric")
        out = solve_bellman(
            (-1, 0, 1), grids, N=3, T=1.0, K=100.0, n_claims=1.0, ell_1=0.0, Gamma_Q=1.0,
            rates=rates, r=0.05, mu=0.05, sigma=0.2, lambda_I=0.0, kappa_I=1.0, eta=0.5,
            kappa_J=1.0, rho=0.0, gamma=0.05, lambda_bar_P=0.025, lambda_bar_T=0.05,
            k_A=k, k_B=k, psi=1.0, epsilon_exec=1e-6,
        )
        _, _, _, spread = reservation_prices(out, grids, 100.0, 0.0, 0.0, 0.0, 0.05, 0.0, 1.0)
        return spread

    assert spread_at(1.0) >= spread_at(0.01) - 1e-6


def test_simulate_deterministic_path_zero_shock_inventory_bounds():
    grids = _grids("geometric")
    rates = np.linspace(-5.0, 5.0, 5)
    out = solve_bellman(
        (1,), grids, N=4, T=1.0, K=100.0, n_claims=1.0, ell_1=0.0, Gamma_Q=1.0,
        rates=rates, r=0.05, mu=0.05, sigma=0.2, lambda_I=0.0, kappa_I=1.0, eta=0.5,
        kappa_J=1.0, rho=0.0, gamma=0.05, lambda_bar_P=0.025, lambda_bar_T=0.05,
        k_A=0.5, k_B=0.5, psi=1.0, epsilon_exec=1e-6, store_policy=True,
    )
    paths = simulate_deterministic_path(out, grids, 100.0, 0.0, 0.0, 0.0, 4, 1.0, 0.05, 0.0, 1.0, 1.0)
    q = np.array(paths[1]["q"])
    assert np.all(np.abs(q) <= 3.0 + 1e-6)
    assert len(paths[1]["t"]) == 5  # N+1 time stamps


@pytest.mark.parametrize("n_a", [5, 9])
def test_make_grids_running_average_bounds_contain_zero(n_a):
    grids = make_grids(**BASE, h_kind="arithmetic", n_s=5, n_i=3, n_q=5, n_j=3, n_a=n_a)
    assert grids.a[0] <= 0.0 <= grids.a[-1]
