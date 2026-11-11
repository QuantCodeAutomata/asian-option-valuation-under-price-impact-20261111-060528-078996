"""Experiment 3: neural-network dealer valuation (Section 5.1, equation (81); Table 2).

Reproduces the neural-network columns of Table 2 ("GPI", "DGM", "Policy") of
"Asian option valuation under price impact" by solving the dealer's reduced
HJB PDE (equation (81)) with two neural-network methods -- the Deep Galerkin
Method (DGM) and Generalized Policy Iteration (GPI) -- instead of the
discrete-time Bellman backward induction of ``src/bellman.py`` (``exp/exp2``).
The GPI policy is additionally evaluated by Monte Carlo simulation of the
discrete-time model (the "Policy" column).

For each of the four Table 2 panels (geometric/arithmetic averaging, with/
without dealer price impact), the three reduced value functions v^theta,
theta in {-1,0,+1}, are each trained ``cfg.NN_TRAIN_RUNS_RUN`` times
independently; since the trained value functions already cover the whole
inventory range Q_bar used by the three reported initial inventories Q0 in
cfg.EXP2_Q0_GRID, training happens once per panel/run (not once per Q0) and
the same nets/policy are evaluated at every Q0 -- this avoids a 3x redundant
retraining that would otherwise be needed only to re-evaluate the same value
function at a different query point.

Run as: python -m exp.exp3_neural_valuation
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config as cfg
from src.neuralnet import reservation_prices_nn, simulate_policy_mc, train_dgm, train_gpi

RESULTS_DIR = "results"
THETAS = (-1, 0, 1)
ASK_SANITY_BOUND = 1_000.0  # beyond this, an ask quote is treated as numerically meaningless


def build_params(lambda_bar_T: float, lambda_bar_P: float) -> dict:
    """Plain dict of market/dealer parameters passed through to src.neuralnet."""
    return dict(
        sigma=cfg.SIGMA, r=cfg.R, mu=cfg.MU, kappa_I=cfg.KAPPA_I, eta=cfg.ETA, rho=cfg.RHO,
        kappa_J=cfg.KAPPA_J, lambda_bar_P=lambda_bar_P, lambda_bar_T=lambda_bar_T,
        k_A=cfg.K_A, k_B=cfg.K_B, psi=cfg.PSI, gamma=cfg.GAMMA, gamma_q=cfg.GAMMA_Q,
        ell_1=cfg.ELL_1, Q_bar=cfg.Q_BAR, nu_bar=cfg.NU_BAR, K=cfg.K, T=cfg.T,
        n_claims=cfg.N_CLAIMS, epsilon_exec=cfg.EPSILON_EXEC, lambda_I=0.0,
    )


def _sanitize_q0_ask(q0: float, bid: float, ask: float, mid: float, spread: float, source: str) -> tuple:
    """Q0=0, lambda_I=rho=0: per Proposition A.2 the seller's reservation ask
    is infinite in continuous time. Report NaN rather than a possibly
    meaningless finite number if the NN estimate is not finite and sane."""
    if q0 == 0.0 and (not np.isfinite(ask) or abs(ask) > ASK_SANITY_BOUND):
        print(f"  note [{source}, Q0=0]: ask not finite/sane ({ask}); reporting NaN (Prop. A.2: infinite ask).")
        return bid, float("nan"), float("nan"), float("nan")
    return bid, ask, mid, spread


def run_dgm_runs(h_kind: str, params: dict, n_runs: int) -> list[dict]:
    """Train n_runs independent sets of 3 DGM nets (theta in {-1,0,1})."""
    nets_per_run = []
    for run in range(n_runs):
        nets = {
            theta: train_dgm(theta, h_kind, params, n_steps=cfg.NN_DGM_TRAIN_STEPS_RUN, seed=cfg.NN_SEED + run)
            for theta in THETAS
        }
        nets_per_run.append(nets)
    return nets_per_run


def run_gpi_runs(h_kind: str, params: dict, n_runs: int) -> tuple[list[dict], list[dict]]:
    """Train n_runs independent sets of 3 GPI (net, policy) pairs."""
    nets_per_run, policies_per_run = [], []
    for run in range(n_runs):
        nets, policies = {}, {}
        for theta in THETAS:
            net, policy = train_gpi(
                theta, h_kind, params, n_outer=cfg.NN_GPI_OUTER_ITERS_RUN,
                n_steps_per_iter=cfg.NN_TRAIN_STEPS_PER_ITER_RUN, seed=cfg.NN_SEED + run,
            )
            nets[theta] = net
            policies[theta] = policy
        nets_per_run.append(nets)
        policies_per_run.append(policies)
    return nets_per_run, policies_per_run


def quotes_across_runs(nets_per_run: list[dict], Q0: float, source: str) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {"bid": [], "ask": [], "mid": [], "spread": []}
    for nets in nets_per_run:
        bid, ask, mid, spread = reservation_prices_nn(nets, cfg.S0, cfg.I0, Q0, 0.0, cfg.R, 0.0, cfg.T)
        bid, ask, mid, spread = _sanitize_q0_ask(Q0, bid, ask, mid, spread, source)
        rows["bid"].append(bid)
        rows["ask"].append(ask)
        rows["mid"].append(mid)
        rows["spread"].append(spread)
    return rows


def summarize_mean_std(rows: dict[str, list[float]]) -> dict[str, float]:
    out = {}
    for key, vals in rows.items():
        arr = np.array(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[f"{key}_mean"] = float(np.mean(finite)) if finite.size > 0 else float("nan")
        out[f"{key}_std_or_se"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
    return out


def main() -> None:
    t0 = time.time()
    panels = [
        ("A: geometric, no impact", "geometric", 0.0, 0.0),
        ("B: geometric, baseline impact", "geometric", cfg.LAMBDA_BAR_T, cfg.LAMBDA_BAR_P),
        ("C: arithmetic, no impact", "arithmetic", 0.0, 0.0),
        ("D: arithmetic, baseline impact", "arithmetic", cfg.LAMBDA_BAR_T, cfg.LAMBDA_BAR_P),
    ]
    rows = []
    for label, h_kind, lbt, lbp in panels:
        params = build_params(lbt, lbp)
        print(f"\n=== Panel {label} ===")

        t_dgm = time.time()
        dgm_nets_runs = run_dgm_runs(h_kind, params, cfg.NN_TRAIN_RUNS_RUN)
        print(f"  DGM training: {time.time() - t_dgm:.1f}s "
              f"({cfg.NN_TRAIN_RUNS_RUN} runs x 3 thetas x {cfg.NN_DGM_TRAIN_STEPS_RUN} steps)")

        t_gpi = time.time()
        gpi_nets_runs, gpi_policies_runs = run_gpi_runs(h_kind, params, cfg.NN_TRAIN_RUNS_RUN)
        print(f"  GPI training: {time.time() - t_gpi:.1f}s "
              f"({cfg.NN_TRAIN_RUNS_RUN} runs x 3 thetas x "
              f"{cfg.NN_GPI_OUTER_ITERS_RUN}x{cfg.NN_TRAIN_STEPS_PER_ITER_RUN} steps)")

        for Q0 in cfg.EXP2_Q0_GRID:
            Q0 = float(Q0)

            dgm_summary = summarize_mean_std(quotes_across_runs(dgm_nets_runs, Q0, "DGM"))
            rows.append({"panel": label, "h_kind": h_kind, "Q0": Q0, "method": "DGM", **dgm_summary})

            gpi_summary = summarize_mean_std(quotes_across_runs(gpi_nets_runs, Q0, "GPI"))
            rows.append({"panel": label, "h_kind": h_kind, "Q0": Q0, "method": "GPI", **gpi_summary})

            policy0 = gpi_policies_runs[0]
            mc = simulate_policy_mc(
                policy0, params, h_kind, Q0=Q0, J0=0.0,
                n_paths=cfg.POLICY_MC_PATHS_RUN, n_steps=cfg.POLICY_MC_STEPS_RUN, seed=cfg.NN_SEED,
            )
            bid_m, bid_se = mc["bid"]
            ask_m, ask_se = mc["ask"]
            mid_m, mid_se = mc["mid"]
            spr_m, spr_se = mc["spread"]
            bid_m, ask_m, mid_m, spr_m = _sanitize_q0_ask(Q0, bid_m, ask_m, mid_m, spr_m, "Policy")
            if not np.isfinite(ask_m):
                ask_se = mid_se = spr_se = float("nan")
            rows.append({
                "panel": label, "h_kind": h_kind, "Q0": Q0, "method": "Policy",
                "bid_mean": bid_m, "bid_std_or_se": bid_se,
                "ask_mean": ask_m, "ask_std_or_se": ask_se,
                "mid_mean": mid_m, "mid_std_or_se": mid_se,
                "spread_mean": spr_m, "spread_std_or_se": spr_se,
            })

    table = pd.DataFrame(rows)
    table.to_csv(f"{RESULTS_DIR}/exp3_table2_nn.csv", index=False)
    print("\nTable 2 (neural-network columns: GPI, DGM, Policy):")
    print(table.round(4).to_string(index=False))
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
