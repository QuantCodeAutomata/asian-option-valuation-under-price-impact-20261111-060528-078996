# Guide for coding agents and contributors

This repository reproduces selected experiments from "Asian option valuation
under price impact": exogenous geometric/arithmetic Asian option pricing
under external order flow (Theorem 3.2 and Monte Carlo), the dealer's
Bellman reservation-price scheme (equations (83)-(88)), two neural-network
approximations of the dealer's HJB equation (81) (Generalized Policy
Iteration and the Deep Galerkin Method), and a comparison of exogenous vs.
dealer valuations. No market data is used; every input is the paper's own
analytical model.

## Layout

- `src/config.py` — every experiment parameter as a module-level UPPER_CASE
  variable, grouped by experiment, each with a one-line source comment (a
  paper section/equation, or "not specified in the paper"). Paper-scale and
  reduced-scale variants are named `*_PAPER` and `*_RUN`.
- `src/pricing.py` — Theorem 3.2 geometric Asian formula, the Kemna-Vorst
  frictionless reference, and Monte Carlo pricing of the arithmetic Asian
  option under the pricing measure `Q`.
- `src/bellman.py` — the discrete dealer Bellman recursion (83)-(88),
  multilinear interpolation of the 5-dimensional continuation value, the
  admissible trading-rate set (87), and reservation bid/ask/spread.
- `src/neuralnet.py` — the GPI and DGM neural-network solvers for the HJB
  equation (81) and the Monte Carlo policy evaluation of a GPI trading rule.
- `data/generate_synthetic.py` — the only place that draws random numbers:
  correlated Gaussian increments (for the Monte Carlo pricers) and the
  discrete `(xi, zeta)` shock pairs (for the Bellman recursion). No market
  data loader is included because this paper requires none.
- `exp/exp1_exogenous_pricing.py` ... `exp/exp4_exogenous_vs_dealer.py` — one
  script per experiment, calling only the functions in `src/`.
- `tests/` — pytest unit tests of `src/` on small, fast inputs.
- `results/` — every table (CSV) and figure (PNG) written by the `exp/`
  scripts, plus `results/RESULTS.md` with the paper comparison.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"              # unit tests (fast)
python -m exp.exp1_exogenous_pricing
python -m exp.exp2_bellman_dealer                  # reduced grid (minutes)
python -m exp.exp2_bellman_dealer --full           # paper-scale grid (hours)
python -m exp.exp3_neural_valuation
python -m exp.exp4_exogenous_vs_dealer
bash reproduce.sh                                   # tests + all four experiments
```

Run from the repository root; `python -m exp.<name>` relies on `src/`,
`data/` and `exp/` being importable packages (each has an `__init__.py`), so
do not edit `sys.path` in code.

## Conventions

- All experiment parameters live only in `src/config.py` (no parameters
  hard-coded inside `src/`, `exp/` or `data/`).
- Data preparation code lives only in `data/`; this paper needs only
  `data/generate_synthetic.py` (no market data, hence no `data_loader.py`).
- Tests live only in `tests/`, use small/fast inputs (short grids, few paths,
  few training steps), and do not depend on a live network call.
- Every run of an `exp/` script overwrites its files in `results/`. The
  numbers and verdicts in `results/RESULTS.md` are not regenerated
  automatically — after a rerun, check them against the new files in
  `results/` and update the Markdown by hand.
- API keys are not applicable to this paper; none are read or required.
