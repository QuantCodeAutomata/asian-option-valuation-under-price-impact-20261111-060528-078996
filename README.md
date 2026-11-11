# Asian option valuation under price impact

Replication of selected experiments from **"Asian option valuation under price
impact"** (paper title and authors as given in the indexed source document).

This repository implements, from the paper's own formulas, two complementary
valuations of Asian (average-price) call options in a market where order flow
and dealer hedging move the underlying price:

1. **Exogenous pricing** (Section 3): a consistent price system in which an
   external order-flow state `I_t` enters the stock's drift under a pricing
   measure `Q`. The geometric Asian option has the closed-form price of
   Theorem 3.2; the arithmetic Asian option is priced by Monte Carlo under
   `Q`.
2. **Dealer pricing** (Section 5): a risk-averse dealer who hedges a short or
   long Asian option position while incurring temporary and permanent price
   impact from her own trading, valued by exponential-utility indifference
   pricing. The dealer's value function solves the discrete Bellman
   recursion of equations (83)-(88), approximated here also by two
   neural-network schemes (Generalized Policy Iteration and the Deep Galerkin
   Method) that target the continuous-time HJB equation (81).

No market data is used: every number in this repository is produced from the
paper's analytical model and Monte Carlo simulation of that model, with all
parameters declared in `src/config.py`.

## Repository layout

```
src/     pricing.py (Theorem 3.2 + arithmetic MC), bellman.py (dealer DP),
         neuralnet.py (GPI / DGM), config.py (all paper parameters)
exp/     exp1_exogenous_pricing.py   -> Table 1, Figure 1(e)-(f)
         exp2_bellman_dealer.py      -> Table 2 (Bellman), Table 4, Table 5, Figure 4
         exp3_neural_valuation.py    -> Table 2 (GPI / DGM / Policy columns)
         exp4_exogenous_vs_dealer.py -> Table 3, Figure 2
data/    generate_synthetic.py: correlated Brownian / discrete shock draws
         used by the Monte Carlo and Bellman simulations (no market data)
tests/   pytest unit tests for src/
results/ CSV tables, PNG figures and RESULTS.md written by the exp/ scripts
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required for this paper.

## Running the tests

```bash
python -m pytest -m "not market_data"
```

(there are no `market_data`-marked tests in this repository; the marker
selector is kept only for consistency with the general test command.)

## Running the experiments

From the repository root:

```bash
python -m exp.exp1_exogenous_pricing
python -m exp.exp2_bellman_dealer          # reduced grid, finishes in minutes
python -m exp.exp2_bellman_dealer --full   # paper-scale grid (see results/RESULTS.md)
python -m exp.exp3_neural_valuation
python -m exp.exp4_exogenous_vs_dealer
```

or simply `bash reproduce.sh`, which runs the tests and every experiment in
this order.

Every run overwrites the corresponding files in `results/`. The numbers and
verdicts quoted in `results/RESULTS.md` were produced by one such run; after a
rerun, check `results/RESULTS.md` against the new files and update it by hand
if anything changed.

## Deviations and compute-time reductions

`src/config.py` keeps, side by side, the paper's own discretization (names
ending in `_PAPER`) and the configuration actually executed (names ending in
`_RUN`) for every experiment whose full grid or training schedule could not
finish within the time available for this replication. See
`results/RESULTS.md`, section "Deviations from the paper", for the complete
list, the estimated running time of the full configuration, and the exact
command that selects it.
