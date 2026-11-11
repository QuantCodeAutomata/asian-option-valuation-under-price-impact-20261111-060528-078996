"""Discrete-time Bellman scheme for the dealer's reservation prices.

Implements the backward induction of Section 5.1, equations (83)-(88): state
transitions for the stock price S_t, order-flow I_t, inventory Q_t, transient-
impact state J_t and running average A_t (R_t for arithmetic, Z_t for
geometric averaging), the admissible trading-rate set U_m(s,q,j) of (87), and
the certainty-equivalent Bellman update (88) under exponential utility.

No optimal-execution or option-pricing library implements this five-state
discrete control problem with execution-price-contingent admissible sets, so
the recursion is implemented from scratch (paper equations (83)-(88)).
scipy is used only for the generic interpolation primitive.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Grids:
    s: np.ndarray
    i: np.ndarray
    q: np.ndarray
    j: np.ndarray
    a: np.ndarray
    h_kind: str  # "arithmetic" (h(s)=s) or "geometric" (h(s)=log s)


def h_func(s: np.ndarray, h_kind: str) -> np.ndarray:
    return s if h_kind == "arithmetic" else np.log(s)


def phi_func(a: np.ndarray, T: float, K: float, h_kind: str) -> np.ndarray:
    """Terminal option payoff Phi(a): (a/T - K)^+ arithmetic, (e^{a/T}-K)^+ geometric."""
    if h_kind == "arithmetic":
        return np.maximum(a / T - K, 0.0)
    return np.maximum(np.exp(a / T) - K, 0.0)


def make_grids(
    S0: float, I0: float, T: float, sigma: float, eta: float, kappa_I: float,
    Q_bar: float, nu_bar: float, kappa_J: float, K: float, h_kind: str,
    n_s: int, n_i: int, n_q: int, n_j: int, n_a: int,
) -> Grids:
    """Build the five state grids (Section 5.1 node counts, Q_t/J_t/R_t/I_t)."""
    s_half_width = max(4.0 * sigma * np.sqrt(T), 0.5)
    s_grid = np.exp(np.linspace(np.log(S0) - s_half_width, np.log(S0) + s_half_width, n_s))

    i_stationary_std = eta / np.sqrt(2.0 * kappa_I) if kappa_I > 1e-6 else eta * np.sqrt(T)
    i_half_width = max(4.0 * i_stationary_std, 2.0 * abs(I0), 1e-3)
    i_grid = np.linspace(I0 - i_half_width, I0 + i_half_width, n_i)

    q_grid = np.linspace(-Q_bar, Q_bar, n_q)

    j_bound = Q_bar + nu_bar / max(kappa_J, 0.1)
    j_grid = np.linspace(-j_bound, j_bound, n_j)

    # A_t = integral_0^t h(S_u) du starts at 0 at t=0 and must stay inside the
    # grid throughout the backward induction; the range spans from 0 to the
    # extreme accumulation h(S_grid)*T, with a small margin for interpolation.
    h_lo, h_hi = h_func(s_grid[0], h_kind), h_func(s_grid[-1], h_kind)
    a_lo, a_hi = min(0.0, h_lo * T), max(0.0, h_hi * T)
    margin = 0.05 * max(a_hi - a_lo, 1.0)
    a_grid = np.linspace(a_lo - margin, a_hi + margin, n_a)

    return Grids(s=s_grid, i=i_grid, q=q_grid, j=j_grid, a=a_grid, h_kind=h_kind)


_CORNERS = list(itertools.product([0, 1], repeat=5))


def interp5d(grids: Grids, values: np.ndarray, qs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Vectorized multilinear interpolation of a 5D grid of ``values`` at ``qs``.

    ``qs`` is a tuple of five arrays of identical shape (query coordinates for
    s, i, q, j, a). Queries outside the grid are clamped to the boundary
    (continuation values are evaluated only a short distance beyond the grid
    edges at this coarse resolution; clamping is a standard, conservative
    extrapolation rule not specified by the paper).
    """
    axes = (grids.s, grids.i, grids.q, grids.j, grids.a)
    idx, w = [], []
    for g, q in zip(axes, qs):
        qc = np.clip(q, g[0], g[-1])
        k = np.clip(np.searchsorted(g, qc, side="right") - 1, 0, len(g) - 2)
        denom = g[k + 1] - g[k]
        wk = np.where(denom > 0, (qc - g[k]) / np.where(denom > 0, denom, 1.0), 0.0)
        idx.append(k)
        w.append(wk)

    result = np.zeros_like(qs[0], dtype=float)
    for corner in _CORNERS:
        corner_idx = tuple(idx[d] + corner[d] for d in range(5))
        weight = np.ones_like(qs[0], dtype=float)
        for d in range(5):
            weight = weight * (w[d] if corner[d] == 1 else (1.0 - w[d]))
        result += weight * values[corner_idx]
    return result


def cost_C(nu: np.ndarray, k_A: float, k_B: float, psi: float) -> np.ndarray:
    """C(nu) = k_A*(nu^+)^{1+psi} + k_B*(nu^-)^{1+psi} (Section 5.1)."""
    return k_A * np.maximum(nu, 0.0) ** (1.0 + psi) + k_B * np.maximum(-nu, 0.0) ** (1.0 + psi)


def cash_outflow_rate(
    s: np.ndarray, q: np.ndarray, j: np.ndarray, nu: np.ndarray,
    lambda_bar_P: float, lambda_bar_T: float, k_A: float, k_B: float, psi: float,
) -> np.ndarray:
    """g(s,q,j,nu) = (s + lambda_bar_P*q + lambda_bar_T*j)*nu + C(nu)."""
    return (s + lambda_bar_P * q + lambda_bar_T * j) * nu + cost_C(nu, k_A, k_B, psi)


def terminal_value(grids: Grids, theta: int, T: float, K: float, n_claims: float,
                    ell_1: float, Gamma_Q: float) -> np.ndarray:
    """v_N^theta(s,i,q,j,a) = q*s - L(q) - theta*n*Phi(a)."""
    S, I, Q, J, A = np.meshgrid(grids.s, grids.i, grids.q, grids.j, grids.a, indexing="ij")
    L = ell_1 * np.abs(Q) + 0.5 * Gamma_Q * Q**2
    Phi = phi_func(A, T, K, grids.h_kind)
    return Q * S - L - theta * n_claims * Phi


def bellman_step(
    v_next: np.ndarray, grids: Grids, t_m: float, dt: float, T: float,
    rates: np.ndarray, r: float, mu: float, sigma: float, lambda_I: float, kappa_I: float,
    eta: float, kappa_J: float, rho: float, gamma: float, lambda_bar_P: float,
    lambda_bar_T: float, k_A: float, k_B: float, psi: float, epsilon_exec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One backward step of the Bellman recursion (88), vectorized over the grid.

    Returns (v_m, nu_opt_m), both with the grid's shape.
    """
    S, I, Q, J, A = np.meshgrid(grids.s, grids.i, grids.q, grids.j, grids.a, indexing="ij")
    shape = S.shape
    n_nodes = S.size
    n_rates = len(rates)

    S_f, I_f, Q_f, J_f, A_f = (x.reshape(-1, 1) for x in (S, I, Q, J, A))
    nu = rates.reshape(1, -1)
    ones = np.ones((n_nodes, n_rates))

    # Admissible rates: (79) one-sided restriction at the inventory bounds and
    # the inventory-change bound |q + nu*dt| <= Q_bar, combined with the
    # execution-price condition (87).
    exec_price = S_f + lambda_bar_P * Q_f + lambda_bar_T * J_f
    price_ok = np.broadcast_to(exec_price >= epsilon_exec, (n_nodes, n_rates))
    q_bar = grids.q[-1]
    at_upper = np.isclose(Q_f, q_bar)
    at_lower = np.isclose(Q_f, grids.q[0])
    rate_ok = np.ones((n_nodes, n_rates), dtype=bool)
    rate_ok &= ~(at_upper & (nu > 0))
    rate_ok &= ~(at_lower & (nu < 0))
    rate_ok &= np.abs(Q_f + nu * dt) <= q_bar + 1e-9
    zero_rate_col = np.argmin(np.abs(rates))
    only_zero = (np.arange(n_rates).reshape(1, -1) == zero_rate_col) & ones.astype(bool)
    admissible = np.where(price_ok, rate_ok, only_zero)

    g_rate = cash_outflow_rate(S_f * ones, Q_f * ones, J_f * ones, nu * ones,
                                lambda_bar_P, lambda_bar_T, k_A, k_B, psi)
    delta_r = (1.0 - np.exp(-r * dt)) / r if abs(r) > 1e-10 else dt
    immediate = -np.exp(r * (T - t_m)) * delta_r * g_rate

    q_next = (Q_f + nu * dt) * ones
    j_next = (J_f + (-kappa_J * J_f + nu) * dt) * ones

    sqrt_dt = np.sqrt(dt)
    shocks = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    exp_sum = np.zeros((n_nodes, n_rates))
    for xi, zeta in shocks:
        p = (1.0 + rho * xi * zeta) / 4.0
        s_next = (S_f * np.exp((mu + lambda_I * I_f - 0.5 * sigma**2) * dt + sigma * sqrt_dt * xi)) * ones
        i_next = (I_f - kappa_I * I_f * dt + eta * sqrt_dt * zeta) * ones
        h_s = h_func(S_f, grids.h_kind) * ones
        h_s_next = h_func(s_next, grids.h_kind)
        a_next = A_f * ones + 0.5 * (h_s + h_s_next) * dt

        v_cont = interp5d(grids, v_next, (s_next, i_next, q_next, j_next, a_next))
        exp_sum += p * np.exp(-gamma * v_cont)

    continuation = -(1.0 / gamma) * np.log(np.maximum(exp_sum, 1e-300))
    value = immediate + continuation
    value = np.where(admissible, value, -np.inf)

    best = np.argmax(value, axis=1)
    v_m = value[np.arange(n_nodes), best].reshape(shape)
    nu_opt = rates[best].reshape(shape)
    return v_m, nu_opt


def solve_bellman(
    theta_list: tuple[int, ...], grids: Grids, N: int, T: float, K: float, n_claims: float,
    ell_1: float, Gamma_Q: float, rates: np.ndarray, r: float, mu: float, sigma: float,
    lambda_I: float, kappa_I: float, eta: float, kappa_J: float, rho: float, gamma: float,
    lambda_bar_P: float, lambda_bar_T: float, k_A: float, k_B: float, psi: float,
    epsilon_exec: float, store_policy: bool = False,
) -> dict:
    """Backward induction of (88) for every theta in ``theta_list``.

    Returns a dict ``{theta: {"v0": array, "policy": [nu_opt_0, ..., nu_opt_{N-1}] or None}}``.
    """
    dt = T / N
    out = {}
    for theta in theta_list:
        v = terminal_value(grids, theta, T, K, n_claims, ell_1, Gamma_Q)
        policy = [] if store_policy else None
        for m in range(N - 1, -1, -1):
            t_m = m * dt
            v, nu_opt = bellman_step(
                v, grids, t_m, dt, T, rates, r, mu, sigma, lambda_I, kappa_I, eta,
                kappa_J, rho, gamma, lambda_bar_P, lambda_bar_T, k_A, k_B, psi, epsilon_exec,
            )
            if store_policy:
                policy.append(nu_opt)
        if store_policy:
            policy.reverse()  # policy[m] is the optimal rate grid at time t_m
        out[theta] = {"v0": v, "policy": policy}
    return out


def reservation_prices(
    v0_by_theta: dict, grids: Grids, S0: float, I0: float, Q0: float, J0: float,
    r: float, t: float, T: float,
) -> tuple[float, float, float, float]:
    """P_ask=e^{-r(T-t)}(v^0-v^{+1}), P_bid=e^{-r(T-t)}(v^{-1}-v^0) (Section 5.1, (76))."""
    a0 = 0.0
    q = (np.array([S0]), np.array([I0]), np.array([Q0]), np.array([J0]), np.array([a0]))
    v_m1 = float(interp5d(grids, v0_by_theta[-1]["v0"], q)[0])
    v_0 = float(interp5d(grids, v0_by_theta[0]["v0"], q)[0])
    v_p1 = float(interp5d(grids, v0_by_theta[1]["v0"], q)[0])
    disc = np.exp(-r * (T - t))
    ask = disc * (v_0 - v_p1)
    bid = disc * (v_m1 - v_0)
    mid = 0.5 * (ask + bid)
    spread = ask - bid
    return bid, ask, mid, spread


def simulate_deterministic_path(
    policy_by_theta: dict, grids: Grids, S0: float, I0: float, Q0: float, J0: float,
    N: int, T: float, mu: float, lambda_I: float, kappa_I: float, kappa_J: float,
) -> dict:
    """Forward path with zero Brownian shocks (xi=zeta=0), Table 5.

    At each date, the trading rate is read off the stored optimal-policy grid
    by interpolation, and the state is advanced using the drift-only part of
    (83)-(85) (zero shock realizations).
    """
    dt = T / N
    results = {}
    for theta, data in policy_by_theta.items():
        policy = data["policy"]
        s, i, q, j, a = S0, I0, Q0, J0, 0.0
        times, rates_path, q_path, j_path = [], [], [], []
        for m in range(N):
            t_m = m * dt
            query = (np.array([s]), np.array([i]), np.array([q]), np.array([j]), np.array([a]))
            nu = float(interp5d(grids, policy[m], query)[0])
            times.append(t_m)
            rates_path.append(nu)
            q_path.append(q)
            j_path.append(j)
            h_s = h_func(np.array(s), grids.h_kind)
            s_next = s * np.exp((mu + lambda_I * i) * dt)
            i_next = i - kappa_I * i * dt
            q_next = q + nu * dt
            j_next = j + (-kappa_J * j + nu) * dt
            a = a + 0.5 * (h_s + h_func(np.array(s_next), grids.h_kind)) * dt
            s, i, q, j = s_next, i_next, q_next, j_next
        times.append(N * dt)
        q_path.append(q)
        j_path.append(j)
        results[theta] = {"t": times, "nu": rates_path, "q": q_path, "j": j_path}
    return results

