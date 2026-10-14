"""Phase 3 methods: constraint-preserving XY-QAOA, warm-start QAOA, penalized
QAOA building blocks, and classical baselines.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

This module provides the *algorithms and simulators* used by the Phase 3
experiments (orchestration lives in :mod:`phase3_experiments`). All model
definitions come from :mod:`phase1_core`; generic optimizer/multistart,
sampling and metric machinery are reused from :mod:`phase2_qaoa`.

Methods
-------
* Method A: constraint-preserving XY-QAOA. Objective-only Hamiltonian
  ``H_C^{(0)}`` on the fixed-weight-``K`` subspace, uniform fixed-weight
  initial state, periodic XY ring mixer. Simulated exactly in the
  ``C(n,K)``-dimensional subspace; validated against the full Qiskit circuit.
* Method B: warm-start QAOA. SLSQP continuous relaxation -> product state
  ``R_y(2 arcsin sqrt(c_i))`` on each qubit, then penalized QAOA with the
  standard X mixer.
* Method C: penalized standard QAOA (``|+>^n``, X mixer, ``H_C(lambda)``).

Classical baselines: deterministic greedy forward selection and feasible-swap
simulated annealing (cardinality preserved by construction).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.quantum_info import SparsePauliOp

import phase1_core as core
import phase2_qaoa as q2

# ---------------------------------------------------------------------------
# Phase 3 constants
# ---------------------------------------------------------------------------
DEPTHS: Tuple[int, ...] = (1, 2, 3)
N_STARTS = 5
# COBYLA cap per start. At the prompt's suggested 100-300 (and at 500) every
# Method A start terminates on the evaluation cap (COBYLA status 3) without
# converging, which under-converges the comparison against Methods B/C. Raised
# to 1500 (matching the Phase 2 rationale) so most starts converge; per-start
# status/nfev and the success fraction are still recorded for every method.
MAXITER = 1500
LAMBDA = 1.0
SHOTS = 4096
PRIMARY_PENALTY_DIAGNOSTIC = (0.1, 1.0, 2.0)
WARM_START_SEED = 31337
SA_RESTARTS = 20


def feasible_indices(n: int, K: int) -> np.ndarray:
    """All basis indices with Hamming weight ``K`` (little-endian convention)."""
    return np.array([s for s in range(1 << n) if bin(s).count("1") == K],
                    dtype=int)


# ---------------------------------------------------------------------------
# Method A: fixed-weight subspace simulator (exact)
# ---------------------------------------------------------------------------
class WeightSubspaceSimulator:
    """Exact evolution inside the fixed-weight-``K`` subspace.

    The objective-only Hamiltonian ``H_C^{(0)}`` is diagonal, and the periodic
    XY ring mixer conserves Hamming weight, so the whole QAOA evolution is
    block-diagonal on the ``C(n,K)``-dimensional subspace. The mixer is
    diagonalised once; each mixer factor is then ``V exp(-i b w) V^T``.
    """

    def __init__(self, n: int, K: int, cost_diag: np.ndarray, reps: int):
        self.n = int(n)
        self.K = int(K)
        self.reps = int(reps)
        self.feasible = feasible_indices(n, K)
        self.dim = int(self.feasible.shape[0])
        self.d_loc = np.asarray(cost_diag, dtype=float)[self.feasible]

        mixer = core.build_xy_ring_mixer(n)
        full = mixer.to_matrix(sparse=True)
        submix = full[np.ix_(self.feasible, self.feasible)].toarray()
        if np.max(np.abs(submix.imag)) > 1e-12:
            raise ValueError("XY ring mixer subspace block is not real")
        dense = np.asarray(submix.real, dtype=float)
        # enforce exact symmetry before diagonalisation
        dense = 0.5 * (dense + dense.T)
        self.mixer_matrix = dense
        self._w, self._V = np.linalg.eigh(dense)

    def statevector(self, theta: Sequence[float]) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (2 * self.reps,):
            raise ValueError(f"theta must have length {2 * self.reps}")
        amp = np.full(self.dim, 1.0 / math.sqrt(self.dim), dtype=complex)
        for k in range(self.reps):
            amp = amp * np.exp(-1j * theta[k] * self.d_loc)
            coeff = self._V.T @ amp
            amp = self._V @ (np.exp(-1j * theta[self.reps + k] * self._w) * coeff)
        return amp

    def energy(self, theta: Sequence[float]) -> float:
        amp = self.statevector(theta)
        return float(np.sum(np.abs(amp) ** 2 * self.d_loc))

    def probabilities(self, theta: Sequence[float]) -> np.ndarray:
        amp = self.statevector(theta)
        probs = np.abs(amp) ** 2
        return probs / probs.sum()

    def prob_dict(self, theta: Sequence[float]) -> Dict[int, float]:
        """Full-basis-index -> probability (all mass on feasible states)."""
        probs = self.probabilities(theta)
        return {int(idx): float(p) for idx, p in zip(self.feasible, probs)}


def build_method_a_circuit(inst: core.PortfolioInstance, cost_op0: SparsePauliOp,
                           reps: int, n: Optional[int] = None) -> q2.QAOACircuit:
    """Qiskit reference circuit for Method A (StatePreparation + XY mixer).

    Uses the installed ``StatePreparation`` over the explicit uniform
    fixed-weight amplitude vector (DickeStatePreparation is unavailable in
    Qiskit 2.x). Kept for validation and circuit-resource reporting; the
    optimization loop uses :class:`WeightSubspaceSimulator`.
    """
    n = inst.n if n is None else n
    gamma = ParameterVector("gamma", reps)
    beta = ParameterVector("beta", reps)
    qc = QuantumCircuit(n, name=f"xy_qaoa_p{reps}")
    qc.append(StatePreparation(core.uniform_weight_state(n, inst.K)), range(n))
    mixer = core.build_xy_ring_mixer(n)
    for k in range(reps):
        qc.append(PauliEvolutionGate(cost_op0, time=gamma[k]), range(n))
        qc.append(PauliEvolutionGate(mixer, time=beta[k]), range(n))
    return q2.QAOACircuit(circuit=qc, gamma=gamma, beta=beta, reps=reps)


# ---------------------------------------------------------------------------
# Full-space simulator for Methods B and C (vectorised, exact)
# ---------------------------------------------------------------------------
_POPCOUNT_CACHE: Dict[int, np.ndarray] = {}


def _popcounts(n: int) -> np.ndarray:
    pc = _POPCOUNT_CACHE.get(n)
    if pc is None:
        pc = np.array([bin(s).count("1") for s in range(1 << n)], dtype=float)
        _POPCOUNT_CACHE[n] = pc
    return pc


def _fwht(a: np.ndarray) -> np.ndarray:
    """In-place unnormalised fast Walsh-Hadamard transform of length ``2^n``.

    Represents ``sqrt(dim) * H^{\\otimes n}`` (each single-qubit Hadamard is
    ``[[1,1],[1,-1]]``, so two applications give ``dim * I``).
    """
    a = np.ascontiguousarray(a, dtype=complex)
    h = 1
    n = a.shape[0]
    while h < n:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h].copy()
        y = a[:, h:].copy()
        a[:, :h] = x + y
        a[:, h:] = x - y
        h *= 2
    return a.reshape(-1)


def _apply_x_mixer(amp: np.ndarray, n: int, beta: float,
                   pc: np.ndarray) -> np.ndarray:
    """Apply ``exp(-i beta sum_i X_i)`` exactly via ``H e^{-i beta Z} H``.

    ``H^{\\otimes n} X_i H^{\\otimes n} = Z_i``, so the X-mixer becomes a
    diagonal phase in the Hadamard basis: entries ``exp(-i beta (n - 2 w))``
    with ``w`` the Hamming weight of the basis index. O(n 2^n) and fully
    vectorised - no per-qubit Python loop.
    """
    dim = 1 << n
    a = _fwht(amp)
    a = a * np.exp(-1j * beta * (n - 2.0 * pc))
    a = _fwht(a)
    return a / dim


def full_statevector(diag: np.ndarray, n: int, reps: int,
                     theta: Sequence[float], init_amp: np.ndarray) -> np.ndarray:
    """Exact evolution of an arbitrary initial amplitude vector.

    Cost evolution is diagonal (``amp *= exp(-i gamma diag)``); the standard X
    mixer ``exp(-i beta sum_i X_i)`` is applied with the fast Walsh-Hadamard
    construction. The little-endian convention ``s = sum_i x_i 2^i`` is used
    throughout. Validated against the Qiskit circuit.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * reps,):
        raise ValueError(f"theta must have length {2 * reps}")
    diag = np.asarray(diag, dtype=float)
    dim = 1 << n
    amp = np.array(init_amp, dtype=complex)
    if amp.shape != (dim,):
        raise ValueError(f"init_amp must have length {dim}")
    pc = _popcounts(n)
    for k in range(reps):
        amp = amp * np.exp(-1j * theta[k] * diag)
        amp = _apply_x_mixer(amp, n, float(theta[reps + k]), pc)
    return amp


class FullSpaceSimulator:
    """Exact finite-dimensional simulator for X-mixer QAOA with a fixed start."""

    def __init__(self, n: int, reps: int, diag: np.ndarray, init_amp: np.ndarray):
        self.n = int(n)
        self.reps = int(reps)
        self.diag = np.asarray(diag, dtype=float)
        self.init_amp = np.asarray(init_amp, dtype=complex)

    def statevector(self, theta: Sequence[float]) -> np.ndarray:
        return full_statevector(self.diag, self.n, self.reps, theta, self.init_amp)

    def energy(self, theta: Sequence[float]) -> float:
        amp = self.statevector(theta)
        return float(np.sum(np.abs(amp) ** 2 * self.diag))

    def probabilities(self, theta: Sequence[float]) -> np.ndarray:
        amp = self.statevector(theta)
        probs = np.abs(amp) ** 2
        return probs / probs.sum()


def uniform_plus_amplitudes(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / math.sqrt(1 << n), dtype=complex)


def product_state_amplitudes(c: Sequence[float]) -> np.ndarray:
    """Amplitude vector of ``prod_i R_y(2 arcsin sqrt(c_i)) |0>``.

    The probability of qubit ``i`` being 1 is exactly ``c_i``. Amplitudes are
    real and positive, so ``amp[s] = sqrt(prod_i (c_i if bit_i(s) else 1-c_i))``
    with the little-endian index ``s = sum_i x_i 2^i``.
    """
    c = np.asarray(c, dtype=float)
    if np.any(c < -1e-12) or np.any(c > 1 + 1e-12):
        raise ValueError("c must lie in [0, 1]")
    c = np.clip(c, 0.0, 1.0)
    n = c.shape[0]
    dim = 1 << n
    bits = ((np.arange(dim)[:, None] >> np.arange(n)) & 1).astype(bool)
    probs = np.prod(np.where(bits, c[None, :], 1.0 - c[None, :]), axis=1)
    return np.sqrt(probs).astype(complex)


# ---------------------------------------------------------------------------
# Method B: continuous relaxation
# ---------------------------------------------------------------------------
def continuous_relaxation(inst: core.PortfolioInstance, lam: float,
                          c0: Optional[Sequence[float]] = None,
                          maxiter: int = 500) -> Dict:
    """SLSQP relaxation of the penalized objective over ``c in [0,1]^n``."""
    from scipy.optimize import minimize

    n, K, q = inst.n, inst.K, inst.q
    Sigma = np.asarray(inst.Sigma, dtype=float)
    mu = np.asarray(inst.mu, dtype=float)
    if c0 is None:
        c0 = np.full(n, K / n)
    c0 = np.asarray(c0, dtype=float)

    def f(c):
        return float(q * c @ Sigma @ c - mu @ c + lam * (c.sum() - K) ** 2)

    def grad(c):
        return 2.0 * q * (Sigma @ c) - mu + 2.0 * lam * (c.sum() - K) * np.ones(n)

    res = minimize(f, c0, method="SLSQP", jac=grad, bounds=[(0.0, 1.0)] * n,
                   options={"maxiter": int(maxiter), "ftol": 1e-12})
    cstar = np.asarray(res.x, dtype=float)
    return {
        "lambda": float(lam),
        "initial_point": [float(v) for v in c0],
        "final_point": [float(v) for v in cstar],
        "objective": float(res.fun),
        "constraint_residual": float(cstar.sum() - K),
        "max_bound_violation": float(max(0.0, -cstar.min(), cstar.max() - 1.0)),
        "num_iterations": int(getattr(res, "nit", -1)),
        "num_evaluations": int(getattr(res, "nfev", -1)),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
    }


def round_relaxation(inst: core.PortfolioInstance,
                     cstar: Sequence[float]) -> Dict:
    """Recover a weight-``K`` portfolio from a relaxed point by rounding+repair.

    Round each ``c_i`` to the nearest integer, then greedily repair the
    cardinality: while too few assets are selected add the unselected asset with
    the largest ``c_i``; while too many are selected drop the selected asset with
    the smallest ``c_i``. Ties break to the smallest index. Returns the portfolio
    and its true unpenalized objective (the relaxation itself is infeasible in
    general, so this repair is a heuristic, not a guarantee).
    """
    c = np.asarray(cstar, dtype=float)
    n, K, q = inst.n, inst.K, inst.q
    x = (c >= 0.5).astype(int)
    order_add = np.argsort(-c, kind="stable")
    order_drop = np.argsort(c, kind="stable")
    while x.sum() < K:
        for i in order_add:
            if x[i] == 0:
                x[i] = 1
                break
    while x.sum() > K:
        for i in order_drop:
            if x[i] == 1:
                x[i] = 0
                break
    final_cost = core.cost_unpenalized(x, inst.mu, inst.Sigma, q)
    return {
        "rounded_assignment": x.tolist(),
        "rounded_asset_indices": sorted(int(i) for i in np.where(x)[0]),
        "objective": float(final_cost),
        "is_feasible": bool(int(x.sum()) == K),
    }


# ---------------------------------------------------------------------------
# Classical baseline 1: greedy forward selection
# ---------------------------------------------------------------------------
def greedy_forward_selection(inst: core.PortfolioInstance) -> Dict:
    """Deterministic greedy: add the asset that most lowers ``C0`` each step.

    Tie-breaking: the smallest asset index. Greedy is a heuristic, not exact.
    """
    n, K, q = inst.n, inst.K, inst.q
    x = np.zeros(n, dtype=int)
    selected: List[int] = []
    trajectory = []
    t0 = time.perf_counter()
    for _ in range(K):
        best_i = None
        best_val = math.inf
        for i in range(n):
            if x[i]:
                continue
            x[i] = 1
            val = core.cost_unpenalized(x, inst.mu, inst.Sigma, q)
            x[i] = 0
            if val < best_val - 1e-15:
                best_val = val
                best_i = i
        x[best_i] = 1
        selected.append(int(best_i))
        trajectory.append({
            "step": len(selected),
            "added_asset": int(best_i),
            "objective": core.cost_unpenalized(x, inst.mu, inst.Sigma, q),
        })
    final_cost = core.cost_unpenalized(x, inst.mu, inst.Sigma, q)
    return {
        "method": "greedy_forward_selection",
        "tie_break": "smallest asset index",
        "asset_indices": sorted(selected),
        "assignment": x.tolist(),
        "objective": float(final_cost),
        "is_feasible": bool(int(x.sum()) == K),
        "trajectory": trajectory,
        "runtime_s": float(time.perf_counter() - t0),
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# Classical baseline 2: feasible-swap simulated annealing
# ---------------------------------------------------------------------------
def simulated_annealing(inst: core.PortfolioInstance, seed: int,
                        n_steps: int, T0: float, T1: float,
                        init: Optional[Sequence[int]] = None) -> Dict:
    """Metropolis simulated annealing that swaps one selected/unselected pair.

    Cardinality ``K`` is preserved by construction. Temperature is geometric
    from ``T0`` to ``T1``. Returns the full run record (best-so-far and final).
    """
    n, K, q = inst.n, inst.K, inst.q
    rng = np.random.default_rng(seed)
    if init is None:
        selected = set(rng.choice(n, size=K, replace=False).tolist())
    else:
        selected = set(int(i) for i in init)
    x = np.zeros(n, dtype=int)
    for i in selected:
        x[i] = 1
    current = core.cost_unpenalized(x, inst.mu, inst.Sigma, q)
    best = current
    best_x = x.copy()
    accepted = 0
    t0 = time.perf_counter()
    temps = T0 * (T1 / T0) ** (np.arange(n_steps) / max(1, n_steps - 1))
    for k in range(n_steps):
        T = float(temps[k])
        s = int(rng.choice(list(selected)))
        u = int(rng.choice([i for i in range(n) if i not in selected]))
        x[s] = 0
        x[u] = 1
        new = core.cost_unpenalized(x, inst.mu, inst.Sigma, q)
        delta = new - current
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            selected.discard(s)
            selected.add(u)
            current = new
            accepted += 1
            if current < best - 1e-15:
                best = current
                best_x = x.copy()
        else:
            x[s] = 1
            x[u] = 0
    runtime = time.perf_counter() - t0
    return {
        "seed": int(seed),
        "n_steps": int(n_steps),
        "T0": float(T0),
        "T1": float(T1),
        "best_cost": float(best),
        "best_assignment": best_x.tolist(),
        "best_asset_indices": sorted(int(i) for i in np.where(best_x)[0]),
        "final_cost": float(current),
        "accepted_moves": int(accepted),
        "acceptance_rate": float(accepted / n_steps),
        "num_evaluations": int(n_steps),
        "runtime_s": float(runtime),
        "is_feasible": bool(int(best_x.sum()) == K),
    }


def run_simulated_annealing(inst: core.PortfolioInstance, n_restarts: int,
                            n_steps: int, T0: float, T1: float,
                            base_seed: int) -> Dict:
    """Run ``n_restarts`` fixed-budget SA restarts and summarise the spread."""
    runs = [simulated_annealing(inst, base_seed + r, n_steps, T0, T1)
            for r in range(n_restarts)]
    bests = np.array([r["best_cost"] for r in runs])
    best_run = min(runs, key=lambda r: r["best_cost"])
    return {
        "method": "feasible_swap_simulated_annealing",
        "n_restarts": int(n_restarts),
        "n_steps_per_restart": int(n_steps),
        "temperature_schedule": "geometric",
        "T0": float(T0),
        "T1": float(T1),
        "base_seed": int(base_seed),
        "restarts": runs,
        "best_cost": float(best_run["best_cost"]),
        "best_asset_indices": best_run["best_asset_indices"],
        "best_assignment": best_run["best_assignment"],
        "mean_best_cost": float(np.mean(bests)),
        "std_best_cost": float(np.std(bests)),
        "min_best_cost": float(np.min(bests)),
        "max_best_cost": float(np.max(bests)),
        "total_runtime_s": float(sum(r["runtime_s"] for r in runs)),
        "total_evaluations": int(sum(r["num_evaluations"] for r in runs)),
    }
