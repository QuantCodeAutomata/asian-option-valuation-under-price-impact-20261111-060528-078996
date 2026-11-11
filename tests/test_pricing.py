"""Unit tests for src/pricing.py (exogenous Asian option pricing, Theorem 3.2)."""

import numpy as np
import pytest

from src.pricing import (
    arithmetic_asian_mc_price,
    geometric_asian_price,
    kemna_vorst_price,
    kernel_K_I,
    simulate_price_and_flow,
)

BASELINE = dict(S0=100.0, K=100.0, T=1.0, sigma=0.20, r=0.05, mu_Q=0.05)
KV_BASELINE = dict(S0=100.0, K=100.0, T=1.0, sigma=0.20, r=0.05)


def test_kemna_vorst_matches_paper_baseline():
    """Section 5.1: the frictionless Kemna-Vorst price at sigma=0.20 is 5.5468."""
    kv = kemna_vorst_price(**KV_BASELINE)
    assert kv == pytest.approx(5.5468, abs=5e-4)


def test_geometric_price_matches_paper_table1_baseline():
    """Table 1: at lambda_I=0.05, kappa_I=1, eta=0.5, rho=I0=0, V_G = 5.5504."""
    vg = geometric_asian_price(
        **BASELINE, lambda_I=0.05, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0
    )
    assert vg == pytest.approx(5.5504, abs=5e-4)
    kv = kemna_vorst_price(**KV_BASELINE)
    imp_pct = 100.0 * (vg - kv) / kv
    assert imp_pct == pytest.approx(0.065, abs=0.01)


def test_geometric_price_reduces_to_kv_at_zero_lambda_I():
    vg0 = geometric_asian_price(
        **BASELINE, lambda_I=0.0, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0
    )
    kv = kemna_vorst_price(**KV_BASELINE)
    assert vg0 == pytest.approx(kv, abs=1e-10)


def test_geometric_price_increases_with_lambda_I_at_positive_I0():
    """Panel B direction: a positive I_0 raises the geometric price (buy pressure)."""
    kwargs = dict(**BASELINE, kappa_I=1.0, eta=0.5, rho=0.0)
    v_pos = geometric_asian_price(**kwargs, lambda_I=0.05, I0=1.0)
    v_zero = geometric_asian_price(**kwargs, lambda_I=0.05, I0=0.0)
    v_neg = geometric_asian_price(**kwargs, lambda_I=0.05, I0=-1.0)
    assert v_pos > v_zero > v_neg


def test_kernel_K_I_is_nonnegative_and_decreasing_in_kappa_I():
    u = np.linspace(0.0, 0.9, 10)
    k_low = kernel_K_I(u, T=1.0, kappa_I=0.5)
    k_high = kernel_K_I(u, T=1.0, kappa_I=5.0)
    assert np.all(k_low >= 0.0) and np.all(k_high >= 0.0)
    assert np.all(k_low >= k_high - 1e-12)


def test_kernel_K_I_kappa_zero_limit_matches_small_kappa():
    u = np.linspace(0.0, 0.9, 10)
    k_tiny = kernel_K_I(u, T=1.0, kappa_I=1e-6)
    k_limit = kernel_K_I(u, T=1.0, kappa_I=1e-9)
    assert np.allclose(k_tiny, k_limit, atol=1e-4)


def test_simulate_price_and_flow_shapes_and_positivity():
    S, I = simulate_price_and_flow(
        S0=100.0, T=1.0, sigma=0.2, mu_Q=0.05, lambda_I=0.05, kappa_I=1.0,
        eta=0.5, rho=0.0, I0=0.0, n_paths=500, n_steps=50, seed=1,
    )
    assert S.shape == (500, 50) and I.shape == (500, 50)
    assert np.all(S > 0.0)
    assert np.allclose(S[:, 0], 100.0)
    assert np.allclose(I[:, 0], 0.0)


def test_arithmetic_mc_price_matches_paper_table1_baseline():
    """Table 1: arithmetic baseline price 5.7686 +/- 0.0016 (2x10^5 paths, 253 steps)."""
    price, hw = arithmetic_asian_mc_price(
        **BASELINE, lambda_I=0.05, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0,
        n_paths=200_000, n_steps=253, seed=20240521,
    )
    assert price == pytest.approx(5.7686, abs=0.03)
    assert 0.0 < hw < 0.01


def test_arithmetic_mc_reference_matches_paper_table1_reference():
    """Table 1: frictionless arithmetic reference (lambda_I=0) is 5.7647 +/- 0.0015."""
    price, hw = arithmetic_asian_mc_price(
        **BASELINE, lambda_I=0.0, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0,
        n_paths=200_000, n_steps=253, seed=20240521,
    )
    assert price == pytest.approx(5.7647, abs=0.03)


def test_arithmetic_price_exceeds_geometric_price():
    """Jensen's inequality: the arithmetic average has higher variance than the
    geometric average, so the arithmetic Asian call is priced above the
    geometric one throughout (paper: 'Arithmetic prices are higher than
    geometric throughout')."""
    geo = geometric_asian_price(
        **BASELINE, lambda_I=0.05, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0,
    )
    arith, _ = arithmetic_asian_mc_price(
        **BASELINE, lambda_I=0.05, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0,
        n_paths=200_000, n_steps=253, seed=20240521,
    )
    assert arith > geo


@pytest.mark.parametrize("n_paths,n_steps", [(10, 2), (100, 5)])
def test_arithmetic_mc_price_runs_on_small_inputs(n_paths, n_steps):
    price, hw = arithmetic_asian_mc_price(
        **BASELINE, lambda_I=0.05, kappa_I=1.0, eta=0.5, rho=0.0, I0=0.0,
        n_paths=n_paths, n_steps=n_steps, seed=1,
    )
    assert price >= 0.0
    assert hw >= 0.0
