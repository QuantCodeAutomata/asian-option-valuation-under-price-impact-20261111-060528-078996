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

    # A_t = integral_0^t h(S_u) du starts at 0 at t=0 and reaches, by t=T, a
    # value concentrated around T*h(S0) with a standard deviation given by the
    # geometric-Asian variance formula v_Z(0) = sigma^2*T^3/3 (Theorem 3.2):
    # centering the grid on the analytic mean/width of the realized integral,
    # instead of on the extreme corner-to-corner range implied by the S grid,
    # keeps resolution where the terminal payoff's curvature matters. Using
    # the extreme-corner range here would spread the same number of nodes
    # over a region orders of magnitude wider than where probability mass
    # actually lies, biasing the interpolated (convex) payoff upward.
    std_log_integral = sigma * np.sqrt(T**3 / 3.0)
    if h_kind == "geometric":
        a_center = T * np.log(S0)
        a_half_width = max(8.0 * std_log_integral, 0.2)
    else:
        a_center = T * S0
        a_half_width = max(8.0 * S0 * std_log_integral, 0.2 * S0)
    a_lo = min(0.0, a_center - a_half_width)
    a_hi = max(a_center + a_half_width, a_center)
    a_grid = np.linspace(a_lo, a_hi, n_a)

    return Grids(s=s_grid, i=i_grid, q=q_grid, j=j_grid, a=a_grid, h_kind=h_kind)


_CORNERS = list(itertools.product([0, 1], repeat=5))


def _interp_nodes_weights(grids: Grids, qs: tuple[np.ndarray, ...]) -> tuple[list, list, tuple]:
    """Lower-corner indices and interpolation weights per dimension, plus grid shape."""
    axes = (grids.s, grids.i, grids.q, grids.j, grids.a)
    shape = tuple(len(g) for g in axes)
    idx, w = [], []
    for g, q in zip(axes, qs):
        qc = np.clip(q, g[0], g[-1])
        k = np.clip(np.searchsorted(g, qc, side="right") - 1, 0, len(g) - 2)
        denom = g[k + 1] - g[k]
        wk = np.where(denom > 0, (qc - g[k]) / np.where(denom > 0, denom, 1.0), 0.0)
        idx.append(k)
        w.append(wk)
    return idx, w, shape


def interp5d(grids: Grids, values: np.ndarray, qs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Vectorized multilinear interpolation of a 5D grid of ``values`` at ``qs``.

    ``qs`` is a tuple of five arrays of identical shape (query coordinates for
    s, i, q, j, a). Queries outside the grid are clamped to the boundary
    (continuation values are evaluated only a short distance beyond the grid
    edges at this coarse resolution; clamping is a standard, conservative
    extrapolation rule not specified by the paper).

    Each of the 32 corners is gathered through a single flattened (raveled)
    index rather than 5-way fancy indexing on the 5D array: computing one
    ``np.ravel_multi_index`` call and a plain ``take`` on the flattened array
    is markedly faster than numpy's advanced indexing with five integer index
    arrays, which matters because this routine runs inside every Bellman
    backward-induction step for every candidate trading rate and shock pair.
    """
    idx, w, shape = _interp_nodes_weights(grids, qs)
    flat_values = values.reshape(-1)
    result = np.zeros_like(qs[0], dtype=float)
    for corner in _CORNERS:
        corner_idx = tuple(idx[d] + corner[d] for d in range(5))
        flat_idx = np.ravel_multi_index(corner_idx, shape)
        weight = np.ones_like(qs[0], dtype=float)
        for d in range(5):
            weight = weight * (w[d] if corner[d] == 1 else (1.0 - w[d]))
        result += weight * flat_values.take(flat_idx)
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


def _interp5d_batch(grids: Grids, values_stack: np.ndarray, qs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Multilinear interpolation of a stack of 5D grids sharing one set of queries.

    ``values_stack`` has shape (B, n_s, n_i, n_q, n_j, n_a); the index/weight
    computation (the expensive part of :func:`interp5d`) is shared across the
    B value functions, giving a speed-up proportional to B over calling
    :func:`interp5d` once per value function (used to batch over theta).
    """
    idx, w, shape = _interp_nodes_weights(grids, qs)
    flat_values = values_stack.reshape(values_stack.shape[0], -1)
    result = np.zeros((values_stack.shape[0],) + qs[0].shape, dtype=float)
    for corner in _CORNERS:
        corner_idx = tuple(idx[d] + corner[d] for d in range(5))
        flat_idx = np.ravel_multi_index(corner_idx, shape)
        weight = np.ones_like(qs[0], dtype=float)
        for d in range(5):
            weight = weight * (w[d] if corner[d] == 1 else (1.0 - w[d]))
        result += weight[None, ...] * flat_values[:, flat_idx]
    return result


def bellman_step_batch(
    v_next_stack: np.ndarray, grids: Grids, t_m: float, dt: float, T: float,
    rates: np.ndarray, r: float, mu: float, sigma: float, lambda_I: float, kappa_I: float,
    eta: float, kappa_J: float, rho: float, gamma: float, lambda_bar_P: float,
    lambda_bar_T: float, k_A: float, k_B: float, psi: float, epsilon_exec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One backward step of the Bellman recursion (88), batched over theta.

    ``v_next_stack`` has shape (B, *grid_shape). Returns (v_m_stack, nu_opt_stack)
    both of shape (B, *grid_shape).

    The maximization over the admissible trading rate is done by looping over
    the (small) set of candidate rates rather than broadcasting a
    (n_nodes, n_rates) array: with the five-dimensional state grid, n_nodes
    already runs into the millions, and multiplying it by n_rates as well as
    by the four shock branches would exceed the memory available for a single
    process. The loop keeps every array at size O(n_nodes) per theta.
    """
    S, I, Q, J, A = np.meshgrid(grids.s, grids.i, grids.q, grids.j, grids.a, indexing="ij")
    shape = S.shape
    B = v_next_stack.shape[0]
    q_bar = grids.q[-1]
    q_lo = grids.q[0]
    delta_r = (1.0 - np.exp(-r * dt)) / r if abs(r) > 1e-10 else dt
    disc_fv = np.exp(r * (T - t_m))
    sqrt_dt = np.sqrt(dt)
    shocks = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    zero_rate_idx = int(np.argmin(np.abs(rates)))

    exec_price = S + lambda_bar_P * Q + lambda_bar_T * J
    price_ok = exec_price >= epsilon_exec
    at_upper = np.isclose(Q, q_bar)
    at_lower = np.isclose(Q, q_lo)

    best_value = np.full((B,) + shape, -np.inf)
    best_nu = np.zeros((B,) + shape)

    for r_idx, nu in enumerate(rates):
        rate_ok = np.ones(shape, dtype=bool)
        if nu > 0:
            rate_ok &= ~at_upper
        elif nu < 0:
            rate_ok &= ~at_lower
        rate_ok &= np.abs(Q + nu * dt) <= q_bar + 1e-9
        admissible = np.where(price_ok, rate_ok, r_idx == zero_rate_idx)

        g_rate = cash_outflow_rate(S, Q, J, np.full(shape, nu), lambda_bar_P, lambda_bar_T, k_A, k_B, psi)
        immediate = -disc_fv * delta_r * g_rate

        q_next = Q + nu * dt
        j_next = J + (-kappa_J * J + nu) * dt

        exp_sum = np.zeros((B,) + shape)
        for xi, zeta in shocks:
            p = (1.0 + rho * xi * zeta) / 4.0
            s_next = S * np.exp((mu + lambda_I * I - 0.5 * sigma**2) * dt + sigma * sqrt_dt * xi)
            i_next = I - kappa_I * I * dt + eta * sqrt_dt * zeta
            h_s = h_func(S, grids.h_kind)
            h_s_next = h_func(s_next, grids.h_kind)
            a_next = A + 0.5 * (h_s + h_s_next) * dt

            v_cont = _interp5d_batch(grids, v_next_stack, (s_next, i_next, q_next, j_next, a_next))
            exp_sum += p * np.exp(-gamma * v_cont)

        continuation = -(1.0 / gamma) * np.log(np.maximum(exp_sum, 1e-300))
        value = immediate[None, ...] + continuation
        value = np.where(admissible[None, ...], value, -np.inf)

        improve = value > best_value
        best_value = np.where(improve, value, best_value)
        best_nu = np.where(improve, nu, best_nu)

    return best_value, best_nu


def solve_bellman(
    theta_list: tuple[int, ...], grids: Grids, N: int, T: float, K: float, n_claims: float,
    ell_1: float, Gamma_Q: float, rates: np.ndarray, r: float, mu: float, sigma: float,
    lambda_I: float, kappa_I: float, eta: float, kappa_J: float, rho: float, gamma: float,
    lambda_bar_P: float, lambda_bar_T: float, k_A: float, k_B: float, psi: float,
    epsilon_exec: float, store_policy: bool = False,
) -> dict:
    """Backward induction of (88) for every theta in ``theta_list``, batched.

    The B=len(theta_list) value functions share identical state transitions
    and admissible sets at every date, so one batched Bellman step replaces
    B separate steps (see :func:`bellman_step_batch`).

    Returns a dict ``{theta: {"v0": array, "policy": [nu_opt_0, ..., nu_opt_{N-1}] or None}}``.
    """
    dt = T / N
    thetas = list(theta_list)
    v_stack = np.stack([terminal_value(grids, th, T, K, n_claims, ell_1, Gamma_Q) for th in thetas])
    policies = {th: ([] if store_policy else None) for th in thetas}
    for m in range(N - 1, -1, -1):
        t_m = m * dt
        v_stack, nu_stack = bellman_step_batch(
            v_stack, grids, t_m, dt, T, rates, r, mu, sigma, lambda_I, kappa_I, eta,
            kappa_J, rho, gamma, lambda_bar_P, lambda_bar_T, k_A, k_B, psi, epsilon_exec,
        )
        if store_policy:
            for b, th in enumerate(thetas):
                policies[th].append(nu_stack[b])
    out = {}
    for b, th in enumerate(thetas):
        policy = policies[th]
        if store_policy:
            policy.reverse()  # policy[m] is the optimal rate grid at time t_m
        out[th] = {"v0": v_stack[b], "policy": policy}
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
        times, rates_path, q_path, j_path, s_path = [], [], [], [], [s]
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
            s_path.append(s)
        times.append(N * dt)
        q_path.append(q)
        j_path.append(j)
        results[theta] = {"t": times, "nu": rates_path, "q": q_path, "j": j_path, "s": s_path}
    return results

