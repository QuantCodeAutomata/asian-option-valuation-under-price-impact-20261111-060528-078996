"""Experiment 4: exogenous vs. dealer price comparison and quote sensitivity.

Reproduces Table 3 and Figure 2 of "Asian option valuation under price
impact": the exogenous Asian option price of Theorem 3.2 (Section 3) and the
dealer's reservation price from the discrete Bellman recursion of Section
5.1, both measured relative to the frictionless reference (Kemna-Vorst for
geometric averaging, Monte Carlo at lambda_I=0 for arithmetic averaging), as
market and dealer parameters are varied panel by panel.

Panels A-D use lambda_I = 0 for the dealer calculation, as specified for
Tables 2 and 4; Panel E varies lambda_I in both valuations. The dealer uses
the reduced Bellman grid of ``src.config`` (N_STEPS_RUN, N_*_NODES_RUN); see
results/RESULTS.md for the paper-scale grid this substitutes for.

Run as: python -m exp.exp4_exogenous_vs_dealer
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src import config as cfg
from src.bellman import make_grids, reservation_prices, solve_bellman
from src.pricing import arithmetic_asian_mc_price, geometric_asian_price, kemna_vorst_price

RESULTS_DIR = "results"


def exogenous_price(h_kind: str, sigma: float, lambda_I: float, kappa_I: float, rho: float = 0.0) -> float:
    """Exogenous price: Theorem 3.2 closed form (geometric) or Q-measure Monte
    Carlo (arithmetic), Section 3.3-3.4."""
    if h_kind == "geometric":
        return geometric_asian_price(
            S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
            lambda_I=lambda_I, kappa_I=kappa_I, eta=cfg.ETA, rho=rho, I0=cfg.I0,
        )
    price, _ = arithmetic_asian_mc_price(
        S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
        lambda_I=lambda_I, kappa_I=kappa_I, eta=cfg.ETA, rho=rho, I0=cfg.I0,
        n_paths=cfg.MC_PATHS_RUN, n_steps=cfg.MC_STEPS_RUN, seed=cfg.MC_SEED,
    )
    return price


def reference_price(h_kind: str, sigma: float) -> float:
    """Frictionless reference at a given sigma (Section 5.1)."""
    if h_kind == "geometric":
        return kemna_vorst_price(S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R)
    price, _ = arithmetic_asian_mc_price(
        S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
        lambda_I=0.0, kappa_I=cfg.KAPPA_I, eta=cfg.ETA, rho=0.0, I0=0.0,
        n_paths=cfg.MC_PATHS_RUN, n_steps=cfg.MC_STEPS_RUN, seed=cfg.MC_SEED,
    )
    return price


def dealer_price(
    h_kind: str, *, sigma: float = cfg.SIGMA, lambda_bar_T: float, lambda_bar_P: float,
    kappa_I: float = cfg.KAPPA_I, kappa_J: float = cfg.KAPPA_J, lambda_I: float = 0.0,
) -> float:
    """Dealer reservation (mid) price at Q0=J0=0, N=N_STEPS_RUN (Section 5.1, eqs 83-88)."""
    grids = make_grids(
        cfg.S0, cfg.I0, cfg.T, sigma, cfg.ETA, kappa_I, cfg.Q_BAR, cfg.NU_BAR,
        kappa_J, cfg.K, h_kind, cfg.N_S_NODES_RUN, cfg.N_I_NODES_RUN,
        cfg.N_Q_NODES_RUN, cfg.N_J_NODES_RUN, cfg.N_A_NODES_RUN,
    )
    rates = np.linspace(-cfg.NU_BAR, cfg.NU_BAR, cfg.N_RATES_RUN)
    out = solve_bellman(
        (-1, 0, 1), grids, cfg.N_STEPS_RUN, cfg.T, cfg.K, cfg.N_CLAIMS, cfg.ELL_1, cfg.GAMMA_Q,
        rates, cfg.R, cfg.MU, sigma, lambda_I, kappa_I, cfg.ETA, kappa_J, cfg.RHO, cfg.GAMMA,
        lambda_bar_P, lambda_bar_T, cfg.K_A, cfg.K_B, cfg.PSI, cfg.EPSILON_EXEC, store_policy=False,
    )
    _, _, mid, _ = reservation_prices(out, grids, cfg.S0, cfg.I0, 0.0, cfg.J0, cfg.R, 0.0, cfg.T)
    return mid


def run_panel(label: str, grid: np.ndarray, x_name: str, params_fn) -> list[dict]:
    """One Table 3 panel: for each grid point and each averaging type, compute
    the exogenous and dealer percentage differences from the reference price."""
    rows = []
    for x in grid:
        p = params_fn(x)
        for h_kind in ("geometric", "arithmetic"):
            sigma = p.get("sigma", cfg.SIGMA)
            ref = reference_price(h_kind, sigma)
            exog = exogenous_price(h_kind, sigma, p.get("exog_lambda_I", 0.0),
                                    p.get("kappa_I", cfg.KAPPA_I), p.get("rho", 0.0))
            dealer = dealer_price(
                h_kind, sigma=sigma, lambda_bar_T=p.get("lambda_bar_T", cfg.LAMBDA_BAR_T),
                lambda_bar_P=p.get("lambda_bar_P", cfg.LAMBDA_BAR_P),
                kappa_I=p.get("kappa_I", cfg.KAPPA_I), kappa_J=p.get("kappa_J", cfg.KAPPA_J),
                lambda_I=p.get("dealer_lambda_I", 0.0),
            )
            rows.append({
                "panel": label, x_name: x, "h_kind": h_kind, "ref": ref,
                "exog_price": exog, "dealer_price": dealer,
                "Exog_pct": 100.0 * (exog - ref) / ref, "Dealer_pct": 100.0 * (dealer - ref) / ref,
            })
    return rows


def run_table3() -> pd.DataFrame:
    rows = []
    rows += run_panel(
        "A: lambda_bar_T", cfg.TABLE3_LAMBDA_T_GRID, "lambda_bar_T",
        lambda x: dict(lambda_bar_T=x, lambda_bar_P=0.0, dealer_lambda_I=0.0, exog_lambda_I=0.0),
    )
    rows += run_panel(
        "B: sigma", cfg.TABLE3_SIGMA_GRID, "sigma",
        lambda x: dict(sigma=x, exog_lambda_I=cfg.LAMBDA_I, dealer_lambda_I=0.0,
                       lambda_bar_T=cfg.LAMBDA_BAR_T, lambda_bar_P=cfg.LAMBDA_BAR_P),
    )
    rows += run_panel(
        "C: kappa", cfg.TABLE3_KAPPA_GRID, "kappa",
        lambda x: dict(kappa_I=x, kappa_J=x, exog_lambda_I=cfg.LAMBDA_I, dealer_lambda_I=0.0,
                       lambda_bar_T=cfg.LAMBDA_BAR_T, lambda_bar_P=cfg.LAMBDA_BAR_P),
    )
    rows += run_panel(
        "D: lambda_bar_P", cfg.TABLE3_LAMBDA_P_GRID, "lambda_bar_P",
        lambda x: dict(lambda_bar_P=x, lambda_bar_T=0.0, dealer_lambda_I=0.0, exog_lambda_I=0.0),
    )
    rows += run_panel(
        "E: lambda_I", cfg.TABLE3_LAMBDA_I_GRID, "lambda_I",
        lambda x: dict(exog_lambda_I=x, dealer_lambda_I=x, lambda_bar_T=0.0, lambda_bar_P=0.0),
    )
    table3 = pd.DataFrame(rows)
    table3.to_csv(f"{RESULTS_DIR}/exp4_table3.csv", index=False)
    return table3


def plot_figure2(table3: pd.DataFrame) -> None:
    panels = table3["panel"].unique()
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    x_names = {"A: lambda_bar_T": "lambda_bar_T", "B: sigma": "sigma", "C: kappa": "kappa",
               "D: lambda_bar_P": "lambda_bar_P", "E: lambda_I": "lambda_I"}
    markers = {"arithmetic": "o", "geometric": "^"}
    for ax, panel in zip(axes[0], panels):
        sub = table3[table3["panel"] == panel]
        x_name = x_names[panel]
        for h_kind, marker in markers.items():
            s = sub[sub["h_kind"] == h_kind].sort_values(x_name)
            ax.plot(s[x_name], s["Exog_pct"], marker=marker, color="tab:blue",
                    label=f"exogenous ({h_kind})")
            ax.plot(s[x_name], s["Dealer_pct"], marker=marker, color="tab:red",
                    label=f"dealer ({h_kind})")
        ax.set_title(panel)
        ax.set_xlabel(x_name)
        ax.set_ylabel("% difference from frictionless reference")
        ax.legend(fontsize=6)
    fig.suptitle("Figure 2: exogenous (blue) vs. dealer (red) price differences from the frictionless reference")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_figure2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    table3 = run_table3()
    print("Table 3 (exogenous vs. dealer, % difference from reference):")
    print(table3.round(4).to_string(index=False))
    plot_figure2(table3)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
