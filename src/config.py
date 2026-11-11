"""Paper-derived parameters for every experiment.

All experiment parameters live here as module-level UPPER_CASE variables.
No classes or dicts are used, so every value can be traced to a single line
and a single source comment. Sources refer to the sections, tables and
equations of "Asian option valuation under price impact".
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared market parameters (Section 5.1, baseline option and market)
# ---------------------------------------------------------------------------
S0 = 100.0          # initial stock price, Section 5.1
K = 100.0           # option strike, Section 5.1 (S_0 = K = 100)
T = 1.0             # maturity in years, Section 5.1
SIGMA = 0.20        # stock return volatility, Section 5.1 baseline
R = 0.05            # risk-free / discount rate, Section 5.1 (r = mu = mu_Q = 0.05)
MU = 0.05           # real-world drift, Section 5.1
MU_Q = 0.05         # drift under the pricing measure Q, Section 5.1
LAMBDA_I = 0.05     # drift loading on order flow, Section 5.1 baseline
KAPPA_I = 1.0       # mean reversion of order flow I_t, Section 5.1 baseline
ETA = 0.5           # volatility of order flow I_t, Section 5.1 baseline
RHO = 0.0           # correlation between dW and dW^I, Section 5.1 baseline
I0 = 0.0            # initial order flow, Section 5.1 baseline

# ---------------------------------------------------------------------------
# Dealer parameters (Section 5.1)
# ---------------------------------------------------------------------------
LAMBDA_BAR_T = 0.05   # transient (temporary) price impact, Section 5.1 baseline
LAMBDA_BAR_P = 0.025  # permanent price impact, Section 5.1 baseline
KAPPA_J = 1.0         # decay rate of transient-impact state J_t, Section 5.1 baseline
K_A = 0.5             # convex cost coefficient, buying, Section 5.1 baseline (k_A=k_B)
K_B = 0.5             # convex cost coefficient, selling, Section 5.1 baseline
PSI = 1.0             # cost-function exponent (C(nu)=k*(nu^+-)^{1+psi}), Section 5.1
GAMMA = 0.05          # exponential-utility risk aversion, Section 5.1 baseline
GAMMA_Q = 1.0         # terminal quadratic inventory penalty coefficient, Section 5.1
ELL_1 = 0.0           # terminal linear inventory penalty coefficient, Section 5.1
Q_BAR = 3.0           # inventory bound, Section 5.1 baseline
NU_BAR = 5.0          # trading-rate bound, Section 5.1 baseline
Q0 = 0.0              # initial inventory, Section 5.1 baseline
J0 = 0.0              # initial transient-impact state, Section 5.1 baseline
N_CLAIMS = 1.0        # number of option claims (n), Section 5.1 baseline
EPSILON_EXEC = 1e-6   # execution-price floor epsilon in (87); not specified in the paper,
                      # set to a small positive number so the condition only disables
                      # trading when the net execution price would turn non-positive

# ---------------------------------------------------------------------------
# Exp 1: exogenous Asian option pricing (Table 1, Figure 1)
# ---------------------------------------------------------------------------
MC_PATHS_PAPER = 200_000   # Monte Carlo paths, Section 5.1 ("2 x 10^5 Monte Carlo paths")
MC_STEPS_PAPER = 253       # trapezoidal nodes incl. S_0 and S_T, Section 5.1 ("253 prices")
MC_PATHS_RUN = 200_000     # executed path count (kept at the paper value: cheap to run)
MC_STEPS_RUN = 253         # executed step count (kept at the paper value)
MC_SEED = 20240521         # reproducibility seed; not specified in the paper

EXP1_LAMBDA_I_GRID = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15])  # Table 1 Panel A
EXP1_I0_GRID = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])           # Table 1 Panel B
EXP1_KAPPA_I_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])        # Table 1 Panel C
EXP1_RHO_GRID = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75])       # Table 1 Panel D
EXP1_FIGURE1_LAMBDA_I_GRID = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15])  # Fig 1(e)-(f)
EXP1_FIGURE1_SIGMA_GRID = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])          # Fig 1(e)-(f)

# ---------------------------------------------------------------------------
# Exp 2: Bellman dealer scheme (Tables 2, 4, 5)
# ---------------------------------------------------------------------------
# Paper-scale discretization (Section 5.1): N=25 trading intervals, 15 equally
# spaced rates in [-5,5], 61/15/121/9 nodes for Q_t/J_t/R_t/I_t. This grid,
# together with the required S_t dimension, is computationally infeasible to
# run to completion within the time available for this replication (see
# results/RESULTS.md); it remains selectable with --full.
N_STEPS_PAPER = 25
N_RATES_PAPER = 15
N_Q_NODES_PAPER = 61
N_J_NODES_PAPER = 15
N_A_NODES_PAPER = 121
N_I_NODES_PAPER = 9
N_S_NODES_PAPER = 41   # not specified in the paper; chosen so the 5D grid in
                       # implementation_notes has a comparable resolution to Q_t

# Reduced executed configuration: same recursion (88), same admissible rate
# set (87), same terminal condition, at a grid coarse enough to finish within
# the time available. See results/RESULTS.md, "Deviations from the paper".
N_STEPS_RUN = 10
N_RATES_RUN = 7
N_Q_NODES_RUN = 9
N_J_NODES_RUN = 3
N_A_NODES_RUN = 41
N_I_NODES_RUN = 3
N_S_NODES_RUN = 9

EXP2_Q0_GRID = np.array([0.0, 1.0, 2.0])  # Table 2 initial inventories

# ---------------------------------------------------------------------------
# Exp 2 / Table 4: dealer cost and risk sensitivity panels
# ---------------------------------------------------------------------------
TABLE4_K_GRID = np.array([0.001, 0.01, 0.1, 0.5, 1.0])              # Panel A
TABLE4_LAMBDA_T_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.3])          # Panel B (lambda_P = lambda_T/2)
TABLE4_LAMBDA_P_GRID = np.array([0.0, 0.025, 0.05, 0.1, 0.2])        # Panel C (lambda_T = 0.05)
TABLE4_KAPPA_J_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0])           # Panel D
TABLE4_PSI_GRID = np.array([0.2, 0.4, 0.6, 0.8, 1.0])                # Panel E
TABLE4_GAMMA_GRID = np.array([0.01, 0.02, 0.05, 0.075, 0.1])         # Panel F

TABLE5_TIME_INDICES_PAPER = [0, 5, 10, 15, 20, 23, 24, 25]  # Table 5 reporting times at N=25
EXTRAPOLATION_N_GRID_PAPER = [25, 50, 100]                  # Table 2 extrapolation N values

# ---------------------------------------------------------------------------
# Exp 3: neural-network dealer valuation (GPI, DGM) -- Table 2 NN columns
# ---------------------------------------------------------------------------
NN_HIDDEN_UNITS = 32          # hidden layer width; not specified in the paper
NN_HIDDEN_LAYERS = 3          # number of hidden layers; not specified in the paper
NN_TRAIN_RUNS_PAPER = 5       # independent training runs averaged, Section 5.1 / Table 2
NN_TRAIN_RUNS_RUN = 2         # reduced to fit the available compute time
NN_GPI_OUTER_ITERS_PAPER = 20  # GPI policy-evaluation / improvement sweeps; not specified
NN_GPI_OUTER_ITERS_RUN = 5
NN_TRAIN_STEPS_PER_ITER_PAPER = 2000  # gradient steps per sweep; not specified
NN_TRAIN_STEPS_PER_ITER_RUN = 150
NN_DGM_TRAIN_STEPS_PAPER = 20000
NN_DGM_TRAIN_STEPS_RUN = 400
NN_COLLOCATION_POINTS = 512   # collocation batch size per gradient step; not specified
NN_LEARNING_RATE = 1e-3       # Adam learning rate; not specified in the paper
POLICY_MC_PATHS_PAPER = 1_000_000  # Section 5.1 ("10^6 simulated paths")
POLICY_MC_PATHS_RUN = 20_000       # reduced to fit the available compute time
POLICY_MC_STEPS_PAPER = 400        # Section 5.1 ("400 trading dates")
POLICY_MC_STEPS_RUN = 50
NN_SEED = 2024

# ---------------------------------------------------------------------------
# Exp 4: exogenous vs dealer comparison (Table 3, Figure 2)
# ---------------------------------------------------------------------------
TABLE3_LAMBDA_T_GRID = np.array([0.0, 0.05, 0.1, 0.15, 0.20])   # Panel A
TABLE3_SIGMA_GRID = np.array([0.1, 0.2, 0.3, 0.4])              # Panel B
TABLE3_KAPPA_GRID = np.array([0.5, 1.0, 2.0, 5.0, 10.0])        # Panel C (kappa_I and kappa_J)
TABLE3_LAMBDA_P_GRID = np.array([0.0, 0.05, 0.1, 0.2])          # Panel D
TABLE3_LAMBDA_I_GRID = np.array([0.0, 0.05, 0.1, 0.15, 0.2])    # Panel E
