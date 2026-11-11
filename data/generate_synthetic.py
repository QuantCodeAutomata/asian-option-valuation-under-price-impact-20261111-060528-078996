"""Synthetic shock generation for the Monte Carlo experiments.

No market data is used anywhere in this replication: every experiment in
"Asian option valuation under price impact" is evaluated on simulated paths
of the price-impact model of Section 2, with parameters fixed analytically
in ``src/config.py``. This module draws the correlated Brownian increments
that drive the stock price S_t and the order-flow state I_t under the
pricing measure Q (Section 3) and under the physical measure P used in the
dealer's Bellman scheme (Section 5.1, equations (83)-(85)).
"""

import numpy as np


def correlated_normals(
    n_paths: int, n_steps: int, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw independent and rho-correlated standard normal increments.

    Returns ``(z1, z2_corr)`` each of shape ``(n_paths, n_steps)`` such that
    ``z1`` drives the stock's Brownian motion and
    ``z2_corr = rho * z1 + sqrt(1 - rho**2) * z2`` drives the order-flow
    Brownian motion, giving ``corr(z1, z2_corr) = rho`` as required by
    ``d<W^Q, W^{I,Q}> = rho * dt``.
    """
    assert n_paths > 0 and n_steps > 0
    assert -1.0 <= rho <= 1.0
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    z2_corr = rho * z1 + np.sqrt(max(0.0, 1.0 - rho**2)) * z2
    return z1, z2_corr


def discrete_shock_pairs(
    n_paths: int, n_steps: int, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the discrete shocks (xi, zeta) in {-1, 1}^2 of equations (83)-(85).

    The joint law is ``p_{xi,zeta} = (1 + rho*xi*zeta) / 4`` (Section 5.1).
    Returns ``(xi, zeta)``, each of shape ``(n_paths, n_steps)`` with values
    in {-1.0, +1.0}.
    """
    assert n_paths > 0 and n_steps > 0
    assert -1.0 <= rho <= 1.0
    rng = np.random.default_rng(seed)
    xi = rng.choice([-1.0, 1.0], size=(n_paths, n_steps))
    # P(zeta = xi) = (1+rho)/2 reproduces p_{xi,zeta} = (1+rho*xi*zeta)/4.
    same_sign = rng.random((n_paths, n_steps)) < (1.0 + rho) / 2.0
    zeta = np.where(same_sign, xi, -xi)
    return xi, zeta
