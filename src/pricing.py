"""Exogenous Asian option pricing under the consistent price system.

Implements the geometric Asian closed-form price of Theorem 3.2 and a Monte
Carlo pricer for the arithmetic Asian option under the pricing measure Q
(Section 3.3), together with the frictionless references (Kemna-Vorst for
geometric averaging, Monte Carlo at lambda_I = 0 for arithmetic averaging).

No standard derivatives-pricing library (QuantLib and its Python bindings
were checked) implements the order-flow-augmented kernel K_I(u) or the
drift/variance adjustments of Theorem 3.2; these equations are therefore
implemented from scratch.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from data.generate_synthetic import correlated_normals


def kernel_K_I(u: np.ndarray | float, T: float, kappa_I: float) -> np.ndarray:
    """Kernel K_I(u) = integral_u^T (T-v) e^{-kappa_I (v-u)} dv (Section 5, p.14).

    Closed form: K_I(u) = (T-u)/kappa_I - (1 - e^{-kappa_I(T-u)}) / kappa_I**2.
    The kappa_I -> 0 limit is handled by a Taylor expansion to avoid 0/0.
    """
    tau = T - np.asarray(u, dtype=float)
    if kappa_I < 1e-8:
        # Taylor limit of the closed form as kappa_I -> 0: tau^2/2.
        return tau**2 / 2.0
    return tau / kappa_I - (1.0 - np.exp(-kappa_I * tau)) / kappa_I**2


def _integral_K_I_squared(T: float, kappa_I: float, n_grid: int = 2000) -> float:
    """integral_0^T K_I(u)^2 du via the trapezoidal rule (Theorem 3.2 variance term)."""
    u = np.linspace(0.0, T, n_grid)
    return float(np.trapz(kernel_K_I(u, T, kappa_I) ** 2, u))


def _integral_TmU_K_I(T: float, kappa_I: float, n_grid: int = 2000) -> float:
    """integral_0^T (T-u) K_I(u) du via the trapezoidal rule (Theorem 3.2 covariance term)."""
    u = np.linspace(0.0, T, n_grid)
    return float(np.trapz((T - u) * kernel_K_I(u, T, kappa_I), u))


def geometric_asian_price(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    mu_Q: float,
    lambda_I: float,
    kappa_I: float,
    eta: float,
    rho: float,
    I0: float,
    t: float = 0.0,
) -> float:
    """Geometric Asian call price under Q (Theorem 3.2, Section 3.4).

    mean m_Z = z + tau*log(s) + (mu_Q - sigma^2/2)*tau^2/2 + lambda_I*i*K_I(t)
    variance v_Z = sigma^2*tau^3/3
                   + lambda_I^2*eta^2*int_t^T K_I(u)^2 du
                   + 2*rho*sigma*lambda_I*eta*int_t^T (T-u)*K_I(u) du
    mu_G = m_Z / T, sigma_G^2 = v_Z / T^2
    V_G = e^{-r*tau} * [e^{mu_G + sigma_G^2/2} * N(d1) - K * N(d2)]
    At t=0, z=0 and log(s)=log(S0).
    """
    assert T > t >= 0.0 or t == 0.0
    assert sigma > 0 and K > 0 and S0 > 0
    tau = T - t
    z0 = 0.0
    m_z = z0 + tau * np.log(S0) + (mu_Q - 0.5 * sigma**2) * tau**2 / 2.0 + lambda_I * I0 * kernel_K_I(t, T, kappa_I)

    var_diffusion = sigma**2 * tau**3 / 3.0
    if abs(lambda_I) < 1e-14:
        v_z = var_diffusion
    else:
        int_k2 = _integral_K_I_squared(T, kappa_I) if t == 0.0 else _integral_K_I_squared_from(t, T, kappa_I)
        int_tmu_k = _integral_TmU_K_I(T, kappa_I) if t == 0.0 else _integral_TmU_K_I_from(t, T, kappa_I)
        v_z = (
            var_diffusion
            + lambda_I**2 * eta**2 * int_k2
            + 2.0 * rho * sigma * lambda_I * eta * int_tmu_k
        )

    mu_G = m_z / T
    sigma_G2 = v_z / T**2
    sigma_G = np.sqrt(max(sigma_G2, 1e-14))
    d1 = (mu_G - np.log(K) + sigma_G2) / sigma_G
    d2 = d1 - sigma_G
    price = np.exp(-r * tau) * (np.exp(mu_G + sigma_G2 / 2.0) * norm.cdf(d1) - K * norm.cdf(d2))
    return float(price)


def _integral_K_I_squared_from(t: float, T: float, kappa_I: float, n_grid: int = 2000) -> float:
    u = np.linspace(t, T, n_grid)
    return float(np.trapz(kernel_K_I(u, T, kappa_I) ** 2, u))


def _integral_TmU_K_I_from(t: float, T: float, kappa_I: float, n_grid: int = 2000) -> float:
    u = np.linspace(t, T, n_grid)
    return float(np.trapz((T - u) * kernel_K_I(u, T, kappa_I), u))


def kemna_vorst_price(S0: float, K: float, T: float, sigma: float, r: float) -> float:
    """Standard Kemna-Vorst geometric Asian price: geometric_asian_price at lambda_I=0."""
    return geometric_asian_price(
        S0=S0, K=K, T=T, sigma=sigma, r=r, mu_Q=r, lambda_I=0.0,
        kappa_I=1.0, eta=0.0, rho=0.0, I0=0.0,
    )


def simulate_price_and_flow(
    S0: float,
    T: float,
    sigma: float,
    mu_Q: float,
    lambda_I: float,
    kappa_I: float,
    eta: float,
    rho: float,
    I0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate S_u and I_u under Q on an equally spaced grid of ``n_steps`` nodes.

    dS/S = (mu_Q + lambda_I*I)*dt + sigma*dW^Q
    dI = -kappa_I*I*dt + eta*dW^{I,Q},  d<W^Q,W^{I,Q}> = rho*dt  (Section 3.3/5.1)

    Returns arrays of shape (n_paths, n_steps) with S and I sampled at the
    ``n_steps`` equally spaced dates including t=0 and t=T (used for the
    trapezoidal rule, Section 5.1: "253 prices, including S_0 and S_T").
    """
    assert n_steps >= 2
    n_intervals = n_steps - 1
    dt = T / n_intervals
    z1, z2 = correlated_normals(n_paths, n_intervals, rho, seed)
    sqrt_dt = np.sqrt(dt)

    S = np.empty((n_paths, n_steps))
    I = np.empty((n_paths, n_steps))
    S[:, 0] = S0
    I[:, 0] = I0
    for m in range(n_intervals):
        s_m, i_m = S[:, m], I[:, m]
        dW = sqrt_dt * z1[:, m]
        dWI = sqrt_dt * z2[:, m]
        S[:, m + 1] = s_m * np.exp((mu_Q + lambda_I * i_m - 0.5 * sigma**2) * dt + sigma * dW)
        I[:, m + 1] = i_m - kappa_I * i_m * dt + eta * dWI
    return S, I


def arithmetic_asian_mc_price(
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    mu_Q: float,
    lambda_I: float,
    kappa_I: float,
    eta: float,
    rho: float,
    I0: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[float, float]:
    """Arithmetic Asian call price via Monte Carlo under Q (Section 3.3, Table 1).

    R_T = integral_0^T S_u du approximated by the trapezoidal rule on
    ``n_steps`` equally spaced nodes including S_0 and S_T. Payoff =
    (R_T/T - K)^+, price = e^{-rT} E^Q[payoff].

    Variance reduction: the geometric-average payoff along the same paths is
    used as a control variate, with its exact mean given by the closed form
    of Theorem 3.2 (the classical Kemna-Vorst control variate; Kemna and
    Vorst (1990) is the reference named by the paper for the frictionless
    geometric price). This is required to reach the Monte Carlo precision
    reported in Section 5.2 (confidence half-widths of a few 10^-3 on 2x10^5
    paths); a plain estimator without a control variate is far less precise
    at that path count.

    Returns (price, 95%-confidence half-width).
    """
    S, I = simulate_price_and_flow(
        S0, T, sigma, mu_Q, lambda_I, kappa_I, eta, rho, I0, n_paths, n_steps, seed
    )
    dt = T / (n_steps - 1)
    weights = np.full(n_steps, dt)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    R_T = S @ weights
    arith_payoff = np.maximum(R_T / T - K, 0.0)

    log_S = np.log(S)
    Z_T = log_S @ weights
    G_T = np.exp(Z_T / T)
    geo_payoff = np.maximum(G_T - K, 0.0)
    geo_price_closed = geometric_asian_price(
        S0=S0, K=K, T=T, sigma=sigma, r=r, mu_Q=mu_Q, lambda_I=lambda_I,
        kappa_I=kappa_I, eta=eta, rho=rho, I0=I0,
    )

    disc = np.exp(-r * T)
    arith_disc = disc * arith_payoff
    geo_disc = disc * geo_payoff
    cov = np.cov(arith_disc, geo_disc, ddof=1)
    b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-14 else 0.0
    cv_sample = arith_disc - b * (geo_disc - geo_price_closed)

    price = float(cv_sample.mean())
    half_width = float(1.96 * cv_sample.std(ddof=1) / np.sqrt(n_paths))
    return price, half_width
