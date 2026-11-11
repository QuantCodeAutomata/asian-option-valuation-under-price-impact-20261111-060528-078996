"""Experiment 1: exogenous Asian option pricing with external order flow.

Reproduces Table 1 (Panels A-D) and Figure 1 (panels e-f) of "Asian option
valuation under price impact": the geometric Asian price from the closed
form of Theorem 3.2, the arithmetic Asian price from Monte Carlo simulation
under the pricing measure Q, and their percentage premia over the
frictionless references (Kemna-Vorst for geometric, Monte Carlo at
lambda_I=0 for arithmetic).

Run as: python -m exp.exp1_exogenous_pricing
"""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from src import config as cfg
from src.pricing import arithmetic_asian_mc_price, geometric_asian_price, kemna_vorst_price

RESULTS_DIR = "results"


def price_pair(lambda_I: float, I0: float, kappa_I: float, rho: float, sigma: float, seed: int):
    """Geometric closed-form and arithmetic Monte Carlo prices at one parameter point."""
    vg = geometric_asian_price(
        S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
        lambda_I=lambda_I, kappa_I=kappa_I, eta=cfg.ETA, rho=rho, I0=I0,
    )
    va, hw = arithmetic_asian_mc_price(
        S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
        lambda_I=lambda_I, kappa_I=kappa_I, eta=cfg.ETA, rho=rho, I0=I0,
        n_paths=cfg.MC_PATHS_RUN, n_steps=cfg.MC_STEPS_RUN, seed=seed,
    )
    return vg, va, hw


def reference_prices(sigma: float):
    """Frictionless references: Kemna-Vorst (geometric) and MC at lambda_I=0 (arithmetic)."""
    kv = kemna_vorst_price(S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R)
    ref_a, ref_hw = arithmetic_asian_mc_price(
        S0=cfg.S0, K=cfg.K, T=cfg.T, sigma=sigma, r=cfg.R, mu_Q=cfg.MU_Q,
        lambda_I=0.0, kappa_I=cfg.KAPPA_I, eta=cfg.ETA, rho=0.0, I0=0.0,
        n_paths=cfg.MC_PATHS_RUN, n_steps=cfg.MC_STEPS_RUN, seed=cfg.MC_SEED,
    )
    return kv, ref_a, ref_hw


def build_panel(label: str, grid, set_params) -> pd.DataFrame:
    """One Table 1 panel: vary a single parameter, holding the rest at baseline."""
    rows = []
    for x in grid:
        params = dict(lambda_I=cfg.LAMBDA_I, I0=cfg.I0, kappa_I=cfg.KAPPA_I, rho=cfg.RHO,
                      sigma=cfg.SIGMA, seed=cfg.MC_SEED)
        params.update(set_params(x))
        kv, ref_a, ref_hw = reference_prices(params["sigma"])
        vg, va, hw = price_pair(params["lambda_I"], params["I0"], params["kappa_I"],
                                 params["rho"], params["sigma"], params["seed"])
        rows.append({
            "panel": label, "x": x,
            "geometric_price": vg, "kv_reference": kv,
            "geometric_imp_pct": 100.0 * (vg - kv) / kv,
            "arithmetic_price": va, "arithmetic_halfwidth": hw,
            "arithmetic_reference": ref_a, "arithmetic_reference_halfwidth": ref_hw,
            "arithmetic_imp_pct": 100.0 * (va - ref_a) / ref_a,
        })
    return pd.DataFrame(rows)


def run_table1() -> pd.DataFrame:
    panel_a = build_panel("A: lambda_I", cfg.EXP1_LAMBDA_I_GRID, lambda x: dict(lambda_I=x))
    panel_b = build_panel("B: I0", cfg.EXP1_I0_GRID, lambda x: dict(I0=x))
    panel_c = build_panel("C: kappa_I", cfg.EXP1_KAPPA_I_GRID, lambda x: dict(kappa_I=x))
    panel_d = build_panel("D: rho", cfg.EXP1_RHO_GRID, lambda x: dict(rho=x))
    table1 = pd.concat([panel_a, panel_b, panel_c, panel_d], ignore_index=True)
    table1.to_csv(f"{RESULTS_DIR}/exp1_table1.csv", index=False)
    return table1


def run_figure1_ef() -> pd.DataFrame:
    """Figure 1(e)-(f): present-value differences as lambda_I and sigma vary jointly."""
    rows = []
    for sigma in cfg.EXP1_FIGURE1_SIGMA_GRID:
        kv, ref_a, _ = reference_prices(sigma)
        for lam in cfg.EXP1_FIGURE1_LAMBDA_I_GRID:
            vg, va, _ = price_pair(lam, cfg.I0, cfg.KAPPA_I, cfg.RHO, sigma, cfg.MC_SEED)
            rows.append({
                "sigma": sigma, "lambda_I": lam,
                "geometric_pv_diff": vg - kv, "arithmetic_pv_diff": va - ref_a,
            })
    fig1 = pd.DataFrame(rows)
    fig1.to_csv(f"{RESULTS_DIR}/exp1_figure1ef.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for sigma in cfg.EXP1_FIGURE1_SIGMA_GRID:
        sub = fig1[fig1["sigma"] == sigma]
        axes[0].plot(sub["lambda_I"], sub["geometric_pv_diff"], marker="o", label=f"sigma={sigma:.2f}")
        axes[1].plot(sub["lambda_I"], sub["arithmetic_pv_diff"], marker="o", label=f"sigma={sigma:.2f}")
    axes[0].set_title("(e) Geometric Asian: value of order flow")
    axes[1].set_title("(f) Arithmetic Asian: value of order flow")
    for ax in axes:
        ax.set_xlabel("drift loading lambda_I")
        ax.axhline(0.0, color="grey", linewidth=0.7)
    axes[0].set_ylabel("Price - frictionless reference")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 1(e)-(f): value of external order flow vs. lambda_I and sigma")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_figure1ef.png", dpi=150)
    plt.close(fig)
    return fig1


def main() -> None:
    table1 = run_table1()
    print(table1.round(4).to_string(index=False))
    fig1 = run_figure1_ef()
    print(f"Figure 1(e)-(f): {len(fig1)} (sigma, lambda_I) points saved.")

    baseline = table1[(table1["panel"] == "A: lambda_I") & (table1["x"] == cfg.LAMBDA_I)]
    print("\nBaseline (Table 1, lambda_I=0.05, I0=rho=0):")
    print(baseline.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
