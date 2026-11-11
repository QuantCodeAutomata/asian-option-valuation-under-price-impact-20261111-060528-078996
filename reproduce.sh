#!/usr/bin/env bash
# Runs the test suite and every experiment of
# "Asian option valuation under price impact" in the configuration described
# in README.md and results/RESULTS.md.
set -u
cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1_exogenous_pricing
python -m exp.exp2_bellman_dealer
python -m exp.exp3_neural_valuation
python -m exp.exp4_exogenous_vs_dealer
