"""Neural-network dealer valuation: DGM and GPI solvers for the HJB PDE.

Implements Section 5.1 / equation (81) (the dealer's HJB PDE for the reduced
value functions v^theta, theta in {-1,0,+1}) with two neural-network
solvers: the Deep Galerkin Method (DGM, Sirignano & Spiliopoulos (2018)) and
Generalized Policy Iteration (GPI), both evaluated with torch.autograd on
sampled collocation points (a standard PINN/DGM implementation -- no
off-the-shelf PDE-solver library handles this five-state control HJB with an
execution-price-contingent admissible set, so both solvers are implemented
from scratch). Also provides a Monte Carlo evaluator of a GPI policy
(``simulate_policy_mc``), matching the discrete-time state transitions of
``src/bellman.py`` and run in path-chunks to respect the memory budget.

Everything here mirrors the corresponding discrete-time definitions in
``src/bellman.py`` (``terminal_value``, ``phi_func``, ``cost_C``,
``cash_outflow_rate``, ``reservation_prices``, admissibility logic in
``bellman_step_batch``) so that the two solution methods (discrete Bellman
recursion vs. continuous-time PDE / neural network) price the same model.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn

from data.generate_synthetic import correlated_normals
from src import config as cfg
from src.bellman import cash_outflow_rate, cost_C, h_func, make_grids  # noqa: F401  (cost_C re-exported for callers)

STATE_KEYS = ("t", "s", "i", "q", "j", "a")

Number = Union[float, int, np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Value function network
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    """MLP mapping (t,s,i,q,j,a) -> scalar v^theta.

    Inputs are normalized internally: ``(x - center) / scale`` for each of
    the six state components, before entering the linear layers. This keeps
    the raw price/inventory scales (S0=100, a~T*S0, etc.) from dominating the
    tanh activations.
    """

    def __init__(self, centers: dict[str, float], scales: dict[str, float]) -> None:
        super().__init__()
        self.centers = dict(centers)
        self.scales = dict(scales)
        layers: list[nn.Module] = []
        in_dim = len(STATE_KEYS)
        for _ in range(cfg.NN_HIDDEN_LAYERS):
            layers.append(nn.Linear(in_dim, cfg.NN_HIDDEN_UNITS))
            layers.append(nn.Tanh())
            in_dim = cfg.NN_HIDDEN_UNITS
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def _norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return (x - self.centers[name]) / self.scales[name]

    def forward(
        self, t: torch.Tensor, s: torch.Tensor, i: torch.Tensor,
        q: torch.Tensor, j: torch.Tensor, a: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [self._norm("t", t), self._norm("s", s), self._norm("i", i),
             self._norm("q", q), self._norm("j", j), self._norm("a", a)],
            dim=1,
        )
        return self.net(x)


# ---------------------------------------------------------------------------
# Sampling box
# ---------------------------------------------------------------------------
def state_bounds(h_kind: str) -> dict[str, tuple[float, float]]:
    """Sampling box for (s,i,q,j,a), reusing src.bellman.make_grids' width logic.

    Only the extreme grid nodes are used (n_* =3 is the minimum valid grid
    size); the interior nodes are irrelevant here since no interpolation is
    performed -- only the box boundaries are read off.
    """
    grids = make_grids(
        cfg.S0, cfg.I0, cfg.T, cfg.SIGMA, cfg.ETA, cfg.KAPPA_I, cfg.Q_BAR, cfg.NU_BAR,
        cfg.KAPPA_J, cfg.K, h_kind, n_s=3, n_i=3, n_q=3, n_j=3, n_a=3,
    )
    return {
        "s": (float(grids.s[0]), float(grids.s[-1])),
        "i": (float(grids.i[0]), float(grids.i[-1])),
        "q": (float(grids.q[0]), float(grids.q[-1])),
        "j": (float(grids.j[0]), float(grids.j[-1])),
        "a": (float(grids.a[0]), float(grids.a[-1])),
    }


def _centers_scales(h_kind: str, T: float) -> tuple[dict[str, float], dict[str, float]]:
    bounds = state_bounds(h_kind)
    centers = {"t": T / 2.0}
    scales = {"t": max(T / 2.0, 1e-3)}
    for key in ("s", "i", "q", "j", "a"):
        lo, hi = bounds[key]
        centers[key] = 0.5 * (lo + hi)
        scales[key] = max(0.5 * (hi - lo), 1e-6)
    return centers, scales


def _sample_collocation(
    h_kind: str, n: int, rng: np.random.Generator, T: float, t_fixed: float | None = None,
) -> dict[str, torch.Tensor]:
    """Sample n collocation points uniformly in the state box (t uniform in
    [0,T], or fixed at t_fixed for terminal-condition points)."""
    bounds = state_bounds(h_kind)

    def _u(lo: float, hi: float) -> np.ndarray:
        return rng.uniform(lo, hi, size=(n, 1)).astype(np.float32)

    if t_fixed is None:
        t_arr = _u(0.0, T)
    else:
        t_arr = np.full((n, 1), t_fixed, dtype=np.float32)
    arrays = (t_arr, _u(*bounds["s"]), _u(*bounds["i"]), _u(*bounds["q"]), _u(*bounds["j"]), _u(*bounds["a"]))
    return {name: torch.tensor(arr, dtype=torch.float32, requires_grad=True) for name, arr in zip(STATE_KEYS, arrays)}


# ---------------------------------------------------------------------------
# Terminal payoff (torch) -- mirrors src.bellman.terminal_value / phi_func
# ---------------------------------------------------------------------------
def _phi_torch(a: torch.Tensor, T: float, K: float, h_kind: str) -> torch.Tensor:
    if h_kind == "arithmetic":
        return torch.clamp(a / T - K, min=0.0)
    return torch.clamp(torch.exp(a / T) - K, min=0.0)


def _h_torch(s: torch.Tensor, h_kind: str) -> torch.Tensor:
    return s if h_kind == "arithmetic" else torch.log(s)


def _terminal_payoff(
    theta: int, h_kind: str, params: dict, s: torch.Tensor, q: torch.Tensor, a: torch.Tensor,
) -> torch.Tensor:
    """v^theta(T,s,i,q,j,a) = q*s - L(q) - theta*n*Phi(a), L(q)=ell_1*|q|+0.5*gamma_q*q^2."""
    L = params["ell_1"] * torch.abs(q) + 0.5 * params["gamma_q"] * q**2
    phi = _phi_torch(a, params["T"], params["K"], h_kind)
    return q * s - L - theta * params["n_claims"] * phi


# ---------------------------------------------------------------------------
# HJB generator and risk term, shared by the max-term (DGM) and fixed-policy
# (GPI policy-evaluation) residuals
# ---------------------------------------------------------------------------
def _generator_and_risk(
    s: torch.Tensor, i: torch.Tensor, v_s: torch.Tensor, v_i: torch.Tensor,
    v_ss: torch.Tensor, v_ii: torch.Tensor, v_si: torch.Tensor, v_a: torch.Tensor,
    h_kind: str, params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma, eta, rho = params["sigma"], params["eta"], params["rho"]
    mu, kappa_I = params["mu"], params["kappa_I"]
    lambda_I = params.get("lambda_I", 0.0)
    h_s = _h_torch(s, h_kind)
    generator = (
        s * (mu + lambda_I * i) * v_s
        - kappa_I * i * v_i
        + h_s * v_a
        + 0.5 * sigma**2 * s**2 * v_ss
        + 0.5 * eta**2 * v_ii
        + rho * sigma * s * eta * v_si
    )
    risk = sigma**2 * s**2 * v_s**2 + 2.0 * rho * sigma * s * eta * v_s * v_i + eta**2 * v_i**2
    return generator, risk


def _first_and_second_derivatives(
    net: ValueNet, t: torch.Tensor, s: torch.Tensor, i: torch.Tensor,
    q: torch.Tensor, j: torch.Tensor, a: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    v = net(t, s, i, q, j, a)
    ones = torch.ones_like(v)
    v_t, v_s, v_i, v_q, v_j, v_a = torch.autograd.grad(
        v, [t, s, i, q, j, a], grad_outputs=ones, create_graph=True,
    )
    v_ss = torch.autograd.grad(v_s, s, grad_outputs=torch.ones_like(v_s), create_graph=True)[0]
    v_si = torch.autograd.grad(v_s, i, grad_outputs=torch.ones_like(v_s), create_graph=True)[0]
    v_ii = torch.autograd.grad(v_i, i, grad_outputs=torch.ones_like(v_i), create_graph=True)[0]
    return v_t, v_s, v_i, v_q, v_j, v_a, v_ss, v_si, v_ii


def _admissible_mask_torch(s: torch.Tensor, q: torch.Tensor, j: torch.Tensor, nu: float, params: dict) -> torch.Tensor:
    """Boolean admissibility mask, mirroring bellman_step_batch's instantaneous
    (dt -> 0) limit: trading disabled whenever the execution price would be
    non-positive; otherwise only trades that do not push |q| past Q_bar are
    allowed."""
    exec_price = s + params["lambda_bar_P"] * q + params["lambda_bar_T"] * j
    price_ok = exec_price >= params["epsilon_exec"]
    q_bar = params["Q_bar"]
    tol = max(1e-6, 1e-4 * q_bar)
    if nu > 0.0:
        rate_ok = q < (q_bar - tol)
    elif nu < 0.0:
        rate_ok = q > (-q_bar + tol)
    else:
        rate_ok = torch.ones_like(q, dtype=torch.bool)
    zero_nu = torch.full_like(q, 1.0 if nu == 0.0 else 0.0, dtype=torch.bool)
    return torch.where(price_ok, rate_ok, zero_nu)


def hjb_residual(
    net: ValueNet, theta: int, h_kind: str,
    t: torch.Tensor, s: torch.Tensor, i: torch.Tensor, q: torch.Tensor,
    j: torch.Tensor, a: torch.Tensor, rates: np.ndarray, params: dict,
) -> torch.Tensor:
    """Per-sample residual of the HJB PDE (81) (batch,1).

    v_t + L^P[v] - kappa_J*j*v_j - (gamma/2)*R[v]
      + max_nu{ nu*(v_q+v_j) - exp(r(T-t))*g(s,q,j,nu) } = 0
    """
    v_t, v_s, v_i, v_q, v_j, v_a, v_ss, v_si, v_ii = _first_and_second_derivatives(net, t, s, i, q, j, a)
    generator, risk = _generator_and_risk(s, i, v_s, v_i, v_ss, v_ii, v_si, v_a, h_kind, params)

    disc = torch.exp(params["r"] * (params["T"] - t))
    best: torch.Tensor | None = None
    for nu in rates:
        nu_f = float(nu)
        admissible = _admissible_mask_torch(s, q, j, nu_f, params)
        g = cash_outflow_rate(
            s, q, j, nu_f, params["lambda_bar_P"], params["lambda_bar_T"], params["k_A"], params["k_B"], params["psi"],
        )
        bracket = nu_f * (v_q + v_j) - disc * g
        bracket = torch.where(admissible, bracket, torch.full_like(bracket, float("-inf")))
        best = bracket if best is None else torch.maximum(best, bracket)
    assert best is not None
    max_term = best

    residual = v_t + generator - params["kappa_J"] * j * v_j - 0.5 * params["gamma"] * risk + max_term
    return residual


def train_dgm(theta: int, h_kind: str, params: dict, n_steps: int, seed: int) -> ValueNet:
    """Deep Galerkin Method training loop for v^theta."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centers, scales = _centers_scales(h_kind, params["T"])
    net = ValueNet(centers, scales)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.NN_LEARNING_RATE)
    rates = np.linspace(-params["nu_bar"], params["nu_bar"], cfg.N_RATES_RUN)

    for _ in range(n_steps):
        optimizer.zero_grad()
        interior = _sample_collocation(h_kind, cfg.NN_COLLOCATION_POINTS, rng, params["T"])
        residual = hjb_residual(
            net, theta, h_kind, interior["t"], interior["s"], interior["i"],
            interior["q"], interior["j"], interior["a"], rates, params,
        )
        interior_loss = torch.mean(residual**2)

        terminal = _sample_collocation(h_kind, cfg.NN_COLLOCATION_POINTS, rng, params["T"], t_fixed=params["T"])
        v_term = net(terminal["t"], terminal["s"], terminal["i"], terminal["q"], terminal["j"], terminal["a"])
        payoff = _terminal_payoff(theta, h_kind, params, terminal["s"], terminal["q"], terminal["a"])
        terminal_loss = torch.mean((v_term - payoff) ** 2)

        loss = interior_loss + terminal_loss
        loss.backward()
        optimizer.step()
    return net


# ---------------------------------------------------------------------------
# Greedy policy and GPI
# ---------------------------------------------------------------------------
def _to_numpy_batch(x: Number, batch: int) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
    return np.broadcast_to(arr.reshape(-1) if arr.ndim > 0 else arr, (batch,)).astype(np.float64)


def _make_policy(net: ValueNet, rates: np.ndarray, params: dict) -> Callable:
    """Greedy policy nu(t,s,i,q,j) w.r.t. the candidate rate set, using v_q, v_j
    from autograd on the current net (state component ``a`` is fixed at 0,
    per the policy-function signature taking only (t,s,i,q,j))."""

    def policy(t: Number, s: Number, i: Number, q: Number, j: Number) -> Union[float, np.ndarray]:
        scalar_in = all(np.ndim(x) == 0 for x in (t, s, i, q, j))
        batch = max(np.size(t), np.size(s), np.size(i), np.size(q), np.size(j))

        t_np = _to_numpy_batch(t, batch)
        s_np = _to_numpy_batch(s, batch)
        i_np = _to_numpy_batch(i, batch)
        q_np = _to_numpy_batch(q, batch)
        j_np = _to_numpy_batch(j, batch)

        t_t = torch.tensor(t_np.reshape(-1, 1), dtype=torch.float32)
        s_t = torch.tensor(s_np.reshape(-1, 1), dtype=torch.float32)
        i_t = torch.tensor(i_np.reshape(-1, 1), dtype=torch.float32)
        q_t = torch.tensor(q_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        j_t = torch.tensor(j_np.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        a_t = torch.zeros_like(t_t)

        # enable_grad() makes this robust to being called from inside a
        # ``torch.no_grad()`` block (e.g. the GPI policy-evaluation residual,
        # which intentionally stop-gradients the policy's *chosen rate* but
        # still needs a local autograd graph to compute that rate from v_q, v_j).
        with torch.enable_grad():
            v = net(t_t, s_t, i_t, q_t, j_t, a_t)
            v_q, v_j = torch.autograd.grad(v, [q_t, j_t], grad_outputs=torch.ones_like(v))
        slope = (v_q + v_j).detach().numpy().reshape(-1)

        disc = np.exp(params["r"] * (params["T"] - t_np))
        exec_price = s_np + params["lambda_bar_P"] * q_np + params["lambda_bar_T"] * j_np
        price_ok = exec_price >= params["epsilon_exec"]
        q_bar = params["Q_bar"]
        tol = max(1e-6, 1e-4 * q_bar)

        best_val = np.full(batch, -np.inf)
        best_rate = np.zeros(batch)
        for nu in rates:
            nu_f = float(nu)
            if nu_f > 0.0:
                rate_ok = q_np < (q_bar - tol)
            elif nu_f < 0.0:
                rate_ok = q_np > (-q_bar + tol)
            else:
                rate_ok = np.ones(batch, dtype=bool)
            admissible = np.where(price_ok, rate_ok, nu_f == 0.0)
            g = cash_outflow_rate(
                s_np, q_np, j_np, nu_f, params["lambda_bar_P"], params["lambda_bar_T"],
                params["k_A"], params["k_B"], params["psi"],
            )
            bracket = nu_f * slope - disc * g
            bracket = np.where(admissible, bracket, -np.inf)
            improve = bracket > best_val
            best_val = np.where(improve, bracket, best_val)
            best_rate = np.where(improve, nu_f, best_rate)

        if scalar_in:
            return float(best_rate[0])
        return best_rate

    return policy


def _policy_residual(
    net: ValueNet, h_kind: str, interior: dict[str, torch.Tensor], params: dict, policy: Callable,
) -> torch.Tensor:
    """One-step discretized Bellman residual under the *current* (stop-gradient)
    policy's rate, replacing the max_nu{...} term of :func:`hjb_residual`."""
    t, s, i, q, j, a = (interior[k] for k in STATE_KEYS)
    v_t, v_s, v_i, v_q, v_j, v_a, v_ss, v_si, v_ii = _first_and_second_derivatives(net, t, s, i, q, j, a)
    generator, risk = _generator_and_risk(s, i, v_s, v_i, v_ss, v_ii, v_si, v_a, h_kind, params)

    with torch.no_grad():
        nu_np = policy(
            t.detach().numpy().reshape(-1), s.detach().numpy().reshape(-1), i.detach().numpy().reshape(-1),
            q.detach().numpy().reshape(-1), j.detach().numpy().reshape(-1),
        )
    nu_t = torch.tensor(np.asarray(nu_np, dtype=np.float32).reshape(-1, 1))

    disc = torch.exp(params["r"] * (params["T"] - t))
    g = cash_outflow_rate(
        s, q, j, nu_t, params["lambda_bar_P"], params["lambda_bar_T"], params["k_A"], params["k_B"], params["psi"],
    )
    bracket = nu_t * (v_q + v_j) - disc * g
    residual = v_t + generator - params["kappa_J"] * j * v_j - 0.5 * params["gamma"] * risk + bracket
    return residual


def train_gpi(
    theta: int, h_kind: str, params: dict, n_outer: int, n_steps_per_iter: int, seed: int,
) -> tuple[ValueNet, Callable]:
    """Generalized Policy Iteration: alternate policy evaluation (fit v to the
    one-step residual under the current policy) and policy improvement
    (recompute the greedy policy from the updated v)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centers, scales = _centers_scales(h_kind, params["T"])
    net = ValueNet(centers, scales)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.NN_LEARNING_RATE)
    rates = np.linspace(-params["nu_bar"], params["nu_bar"], cfg.N_RATES_RUN)

    policy = _make_policy(net, rates, params)
    for _ in range(n_outer):
        for _ in range(n_steps_per_iter):
            optimizer.zero_grad()
            interior = _sample_collocation(h_kind, cfg.NN_COLLOCATION_POINTS, rng, params["T"])
            residual = _policy_residual(net, h_kind, interior, params, policy)
            interior_loss = torch.mean(residual**2)

            terminal = _sample_collocation(h_kind, cfg.NN_COLLOCATION_POINTS, rng, params["T"], t_fixed=params["T"])
            v_term = net(terminal["t"], terminal["s"], terminal["i"], terminal["q"], terminal["j"], terminal["a"])
            payoff = _terminal_payoff(theta, h_kind, params, terminal["s"], terminal["q"], terminal["a"])
            terminal_loss = torch.mean((v_term - payoff) ** 2)

            loss = interior_loss + terminal_loss
            loss.backward()
            optimizer.step()
        policy = _make_policy(net, rates, params)
    return net, policy


# ---------------------------------------------------------------------------
# Reservation prices from trained nets
# ---------------------------------------------------------------------------
def reservation_prices_nn(
    nets: dict[int, ValueNet], S0: float, I0: float, Q0: float, J0: float, r: float, t: float, T: float,
) -> tuple[float, float, float, float]:
    """Same formula/order as src.bellman.reservation_prices: (bid, ask, mid, spread)."""
    with torch.no_grad():
        t_t = torch.tensor([[t]], dtype=torch.float32)
        s_t = torch.tensor([[S0]], dtype=torch.float32)
        i_t = torch.tensor([[I0]], dtype=torch.float32)
        q_t = torch.tensor([[Q0]], dtype=torch.float32)
        j_t = torch.tensor([[J0]], dtype=torch.float32)
        a_t = torch.tensor([[0.0]], dtype=torch.float32)
        v_m1 = float(nets[-1](t_t, s_t, i_t, q_t, j_t, a_t).item())
        v_0 = float(nets[0](t_t, s_t, i_t, q_t, j_t, a_t).item())
        v_p1 = float(nets[1](t_t, s_t, i_t, q_t, j_t, a_t).item())
    disc = float(np.exp(-r * (T - t)))
    ask = disc * (v_0 - v_p1)
    bid = disc * (v_m1 - v_0)
    mid = 0.5 * (ask + bid)
    spread = ask - bid
    return bid, ask, mid, spread


# ---------------------------------------------------------------------------
# Monte Carlo policy evaluation (Table 2 'Policy' column)
# ---------------------------------------------------------------------------
def simulate_policy_mc(
    policy_by_theta: dict[int, Callable], params: dict, h_kind: str,
    Q0: float, J0: float, n_paths: int, n_steps: int, seed: int,
) -> dict[str, tuple[float, float]]:
    """Monte Carlo evaluation of a GPI policy under the discrete-time model of
    src/bellman.py. Processes paths in chunks of at most 20_000 to respect
    the memory budget, accumulating batch certainty-equivalent estimates
    (so the standard error can be estimated across batches)."""
    chunk_size = 20_000
    thetas = (-1, 0, 1)
    n_batches = 20
    batch_paths = max(1, n_paths // n_batches)
    dt = params["T"] / n_steps
    delta_r = (1.0 - np.exp(-params["r"] * dt)) / params["r"] if abs(params["r"]) > 1e-10 else dt
    disc_r = np.exp(params["r"] * dt)

    v_hat_batches: dict[int, list[float]] = {th: [] for th in thetas}

    remaining = n_paths
    batch_idx = 0
    while remaining > 0:
        this_batch = min(batch_paths, remaining)
        remaining -= this_batch
        sub_remaining = this_batch
        # accumulate per-theta exp(-gamma*W_T) sums over sub-chunks of <=20_000 paths
        sums = {th: 0.0 for th in thetas}
        count = 0
        chunk_seed_base = seed + 1000 * batch_idx
        sub_idx = 0
        while sub_remaining > 0:
            n_chunk = min(chunk_size, sub_remaining)
            sub_remaining -= n_chunk
            xi, zeta = correlated_normals(n_chunk, n_steps, params["rho"], chunk_seed_base + sub_idx)
            sub_idx += 1

            s = np.full(n_chunk, cfg.S0)
            i_state = np.full(n_chunk, cfg.I0)
            q_by_theta = {th: np.full(n_chunk, Q0) for th in thetas}
            j_by_theta = {th: np.full(n_chunk, J0) for th in thetas}
            a_by_theta = {th: np.zeros(n_chunk) for th in thetas}
            x_by_theta = {th: np.zeros(n_chunk) for th in thetas}

            sqrt_dt = np.sqrt(dt)
            for m in range(n_steps):
                t_m = m * dt
                s_next = s * np.exp(
                    (params["mu"] + params.get("lambda_I", 0.0) * i_state - 0.5 * params["sigma"] ** 2) * dt
                    + params["sigma"] * sqrt_dt * xi[:, m]
                )
                i_next = i_state - params["kappa_I"] * i_state * dt + params["eta"] * sqrt_dt * zeta[:, m]
                h_s = h_func(s, h_kind)
                h_s_next = h_func(s_next, h_kind)

                for th in thetas:
                    q = q_by_theta[th]
                    j = j_by_theta[th]
                    nu = policy_by_theta[th](np.full(n_chunk, t_m), s, i_state, q, j)
                    nu = np.asarray(nu, dtype=np.float64).reshape(-1)
                    g = cash_outflow_rate(
                        s, q, j, nu, params["lambda_bar_P"], params["lambda_bar_T"],
                        params["k_A"], params["k_B"], params["psi"],
                    )
                    x_by_theta[th] = disc_r * (x_by_theta[th] - delta_r * g)
                    q_by_theta[th] = q + nu * dt
                    j_by_theta[th] = j + (-params["kappa_J"] * j + nu) * dt
                    a_by_theta[th] = a_by_theta[th] + 0.5 * (h_s + h_s_next) * dt

                s, i_state = s_next, i_next

            for th in thetas:
                L = params["ell_1"] * np.abs(q_by_theta[th]) + 0.5 * params["gamma_q"] * q_by_theta[th] ** 2
                if h_kind == "arithmetic":
                    phi = np.maximum(a_by_theta[th] / params["T"] - params["K"], 0.0)
                else:
                    phi = np.maximum(np.exp(a_by_theta[th] / params["T"]) - params["K"], 0.0)
                w_T = x_by_theta[th] + q_by_theta[th] * s - L - th * params["n_claims"] * phi
                sums[th] += np.sum(np.exp(-params["gamma"] * w_T))
            count += n_chunk

        for th in thetas:
            mean_exp = sums[th] / count
            v_hat = -(1.0 / params["gamma"]) * np.log(max(mean_exp, 1e-300))
            v_hat_batches[th].append(v_hat)
        batch_idx += 1

    v_hat_mean = {th: float(np.mean(v_hat_batches[th])) for th in thetas}
    v_hat_se = {
        th: float(np.std(v_hat_batches[th], ddof=1) / np.sqrt(len(v_hat_batches[th]))) if len(v_hat_batches[th]) > 1 else 0.0
        for th in thetas
    }

    disc = np.exp(-params["r"] * params["T"])

    def _combo(coeffs: dict[int, float]) -> tuple[float, float]:
        mean = disc * sum(coeffs[th] * v_hat_mean[th] for th in coeffs)
        var = sum((disc * coeffs[th]) ** 2 * v_hat_se[th] ** 2 for th in coeffs)
        return mean, float(np.sqrt(var))

    bid_mean, bid_se = _combo({-1: 1.0, 0: -1.0})
    ask_mean, ask_se = _combo({0: 1.0, 1: -1.0})
    mid_mean = 0.5 * (bid_mean + ask_mean)
    mid_se = 0.5 * np.sqrt(bid_se**2 + ask_se**2)
    spread_mean = ask_mean - bid_mean
    spread_se = np.sqrt(bid_se**2 + ask_se**2)

    return {
        "bid": (bid_mean, bid_se),
        "ask": (ask_mean, ask_se),
        "mid": (mid_mean, mid_se),
        "spread": (spread_mean, spread_se),
    }
