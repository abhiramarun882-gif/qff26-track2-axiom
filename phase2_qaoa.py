"""Phase 2 core: standard penalized QAOA baseline for the Beginner instance.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

Scope (Phase 2 only)
--------------------
* Standard transverse-field QAOA with the penalized cost Hamiltonian
  ``H_C(lambda)`` and mixer ``H_M = sum_i X_i``, initial state ``|+>^n``.
* Depth study (p = 1..4) and penalty-coefficient study
  (lambda in {0.05, 0.1, 0.25, 0.5, 1, 2, 5}) on the Beginner instance
  (n = 6, K = 3, q = 0.5, seed = 2026).
* Multistart COBYLA optimization, exact statevector energy, fixed-seed
  sampling, metric definitions, exact penalty analysis, plotting.

NOT in scope here: learned angle prediction, ML training, Intermediate
XY-QAOA, warm-start QAOA, noise/readout/hardware.

Conventions
-----------
All model conventions (asset i <-> qubit i, little-endian index
``s = sum_i x_i 2^i``, ``x_i = (I - Z_i)/2``) are inherited from
``phase1_core`` and are NOT redefined here.

Parameter vector convention (explicit circuit):
    theta = [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
which is exactly the order of ``circuit.parameters`` for the circuit built by
:func:`build_qaoa_circuit`. Note this differs from ``QAOAAnsatz``, whose order
is ``[beta..., gamma...]``; :func:`qaoa_ansatz_energy` handles the reordering.
Layer sequence per repetition (chronological): cost evolution ``e^{-i gamma H_C}``
then mixer ``e^{-i beta H_M}``, matching
``prod_k e^{-i beta_k H_M} e^{-i gamma_k H_C}`` with the rightmost acting first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

import phase1_core as core

# ---------------------------------------------------------------------------
# Phase 2 constants
# ---------------------------------------------------------------------------
ENERGY_ATOL = 1e-10        # exact vs independently computed expectation
PROB_ATOL = 1e-12          # probability normalisation
CONVERGENCE_ATOL = 1e-8    # optimizer "converged" energy stability

DEPTHS: Tuple[int, ...] = (1, 2, 3, 4)
PENALTIES: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
PRIMARY_LAMBDA = 1.0
PRIMARY_DEPTH = 2
SHOTS = 4096
OPTIMIZER_NAME = "scipy.optimize.minimize:COBYLA"
DEFAULT_MAXITER = 250
DEFAULT_RHOBEG = 1.0


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def build_x_mixer(n: int) -> SparsePauliOp:
    """Standard transverse-field mixer ``H_M = sum_{i=0}^{n-1} X_i``.

    Qiskit Pauli string convention: rightmost character is qubit 0, so an ``X``
    on qubit ``i`` sits at string position ``n - 1 - i``.
    """
    if n < 1:
        raise ValueError(f"mixer requires n >= 1, got {n}")
    terms: List[Tuple[str, float]] = []
    for i in range(n):
        chars = ["I"] * n
        chars[n - 1 - i] = "X"
        terms.append(("".join(chars), 1.0))
    return SparsePauliOp.from_list(terms)


def cost_operator(inst: core.PortfolioInstance, lam: float) -> SparsePauliOp:
    """Validated penalized cost operator ``H_C(lambda)`` from Phase 1."""
    op, _, _, _ = core.qubo_to_pauli(inst.mu, inst.Sigma, inst.q, inst.K, lam)
    return op


# ---------------------------------------------------------------------------
# Explicit parameterized QAOA circuit
# ---------------------------------------------------------------------------
@dataclass
class QAOACircuit:
    """A QAOA circuit plus its explicit parameter ordering."""

    circuit: QuantumCircuit
    gamma: ParameterVector
    beta: ParameterVector
    reps: int

    @property
    def theta_order(self) -> List:
        """Parameter objects in the order expected by ``theta`` vectors."""
        return list(self.gamma) + list(self.beta)

    @property
    def num_parameters(self) -> int:
        return 2 * self.reps

    def split_theta(self, theta: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Split a flat ``theta`` (gammas then betas) into (gamma, beta)."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.num_parameters,):
            raise ValueError(f"theta must have length {self.num_parameters}, "
                             f"got {theta.shape}")
        return theta[: self.reps].copy(), theta[self.reps:].copy()


def build_qaoa_circuit(n: int, reps: int, cost_op: SparsePauliOp,
                       mixer_op: Optional[SparsePauliOp] = None) -> QAOACircuit:
    """Build the explicit standard QAOA circuit.

    Applies ``H`` on every qubit, then for each repetition:

        cost evolution  e^{-i gamma_k H_C}   (PauliEvolutionGate, time=gamma_k)
        mixer evolution e^{-i beta_k H_M}    (rx(2 beta_k) per qubit, since
                                              e^{-i beta X} = RX(2 beta))

    The mixer is implemented as ``rx(2 beta)`` gates, which is exact and equals
    the operator ``e^{-i beta sum_i X_i}``. ``mixer_op`` is accepted only to
    cross-check (its presence is asserted to equal ``sum_i X_i``).
    """
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    if mixer_op is not None:
        expected = build_x_mixer(n)
        if not _operators_close(mixer_op, expected, atol=1e-12):
            raise ValueError("mixer_op is not the transverse-field sum_i X_i mixer")

    gamma = ParameterVector("gamma", reps)
    beta = ParameterVector("beta", reps)

    qc = QuantumCircuit(n, name=f"qaoa_p{reps}")
    qc.h(range(n))
    for k in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=gamma[k]), range(n))
        for i in range(n):
            qc.rx(2.0 * beta[k], i)
    return QAOACircuit(circuit=qc, gamma=gamma, beta=beta, reps=reps)


def _operators_close(a: SparsePauliOp, b: SparsePauliOp, atol: float = 1e-12) -> bool:
    diff = (a - b).simplify()
    if diff.size == 0:
        return True
    return float(np.max(np.abs(diff.coeffs))) <= atol


def bind_theta(qc: QAOACircuit, theta: Sequence[float]) -> QuantumCircuit:
    """Assign a flat ``theta`` (gammas then betas) to the circuit parameters."""
    gammas, betas = qc.split_theta(theta)
    mapping = {p: float(v) for p, v in zip(qc.gamma, gammas)}
    mapping.update({p: float(v) for p, v in zip(qc.beta, betas)})
    return qc.circuit.assign_parameters(mapping)


# ---------------------------------------------------------------------------
# Exact energy evaluation
# ---------------------------------------------------------------------------
def qaoa_statevector(qc: QAOACircuit, theta: Sequence[float]) -> Statevector:
    """Exact statevector of the bound QAOA circuit."""
    return Statevector(bind_theta(qc, theta))


def qaoa_energy(qc: QAOACircuit, cost_op: SparsePauliOp,
                theta: Sequence[float]) -> float:
    """Exact ``<psi(theta)| H_C |psi(theta)>`` via a Statevector."""
    return float(qaoa_statevector(qc, theta).expectation_value(cost_op).real)


def qaoa_energy_from_vector(cost_op: SparsePauliOp, vec: np.ndarray) -> float:
    """Exact energy computed by an independent diagonal contraction.

    For a diagonal ``H_C`` this equals ``sum_s |psi_s|^2 * diag_s`` and uses
    :func:`phase1_core.operator_diagonal`; it shares no statevector machinery
    with :func:`qaoa_energy` (used for cross-validation only).
    """
    diag = core.operator_diagonal(cost_op)
    probs = np.abs(np.asarray(vec)) ** 2
    return float(np.sum(probs * diag))


def fast_statevector(n: int, reps: int, diag: np.ndarray,
                     theta: Sequence[float]) -> np.ndarray:
    """Exact statevector via a direct, matrix-free simulation.

    The cost Hamiltonian is diagonal, so ``e^{-i gamma H_C}`` multiplies each
    amplitude by ``exp(-i gamma diag[s])``; the mixer ``e^{-i beta X_i}`` is the
    two-level rotation ``cos(beta) I - i sin(beta) X`` on each qubit. This
    reproduces the Qiskit circuit exactly (validated to ~1e-12) but is ~3 orders
    of magnitude faster, so it is used for the optimizer inner loop while the
    Qiskit circuit remains the validated reference and sampling path.

    Parameter order is ``theta = [gammas..., betas...]``; the chronological
    layer order is cost then mixer.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (2 * reps,):
        raise ValueError(f"theta must have length {2 * reps}, got {theta.shape}")
    diag = np.asarray(diag, dtype=float)
    dim = 2 ** n
    if diag.shape != (dim,):
        raise ValueError(f"diag must have length {dim}, got {diag.shape}")

    amp = np.full(dim, 1.0 / math.sqrt(dim), dtype=complex)
    for k in range(reps):
        amp = amp * np.exp(-1j * theta[k] * diag)
        c = math.cos(theta[reps + k])
        s = math.sin(theta[reps + k])
        for i in range(n):
            bit = 1 << i
            for base in range(dim):
                if base & bit:
                    continue
                a0 = amp[base]
                a1 = amp[base | bit]
                amp[base] = c * a0 - 1j * s * a1
                amp[base | bit] = -1j * s * a0 + c * a1
    return amp


class FastQAOASimulator:
    """Exact fast simulator for the standard QAOA energy and distribution."""

    def __init__(self, n: int, reps: int, diag: np.ndarray):
        self.n = int(n)
        self.reps = int(reps)
        self.diag = np.asarray(diag, dtype=float)

    @classmethod
    def from_cost_operator(cls, n: int, reps: int,
                           cost_op: SparsePauliOp) -> "FastQAOASimulator":
        return cls(n, reps, core.operator_diagonal(cost_op))

    def statevector(self, theta: Sequence[float]) -> np.ndarray:
        return fast_statevector(self.n, self.reps, self.diag, theta)

    def energy(self, theta: Sequence[float]) -> float:
        amp = self.statevector(theta)
        return float(np.sum(np.abs(amp) ** 2 * self.diag))

    def probabilities(self, theta: Sequence[float]) -> np.ndarray:
        amp = self.statevector(theta)
        probs = np.abs(amp) ** 2
        total = probs.sum()
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"probabilities do not sum to 1 (got {total})")
        return probs / total


def qaoa_ansatz_energy(n: int, reps: int, cost_op: SparsePauliOp,
                       theta: Sequence[float]) -> float:
    """Energy via Qiskit's ``QAOAAnsatz`` (independent implementation).

    ``QAOAAnsatz`` orders parameters ``[beta..., gamma...]``; ``theta`` here is
    ``[gamma..., beta...]``, so it is reordered before assignment.
    """
    from qiskit.circuit.library import QAOAAnsatz

    init = QuantumCircuit(n)
    init.h(range(n))
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps,
                        initial_state=init, mixer_operator=build_x_mixer(n))
    gammas, betas = np.asarray(theta, float)[:reps], np.asarray(theta, float)[reps:]
    ordered = list(betas) + list(gammas)
    bound = ansatz.assign_parameters(ordered)
    return float(Statevector(bound).expectation_value(cost_op).real)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
@dataclass
class StartResult:
    """One multistart optimization record."""

    start_index: int
    x0: List[float]
    x_final: List[float]
    energy_final: float
    nfev: int
    success: bool
    status: int
    message: str
    elapsed_s: float


def random_initial_parameters(rng: np.random.Generator, reps: int
                              ) -> np.ndarray:
    """Initial ``theta = [gammas..., betas...]``.

    Gammas are drawn uniform in ``[0, 2 pi)`` and betas uniform in ``[0, pi)``
    (reflecting the periodicity of the respective evolutions).
    """
    gammas = rng.uniform(0.0, 2.0 * math.pi, size=reps)
    betas = rng.uniform(0.0, math.pi, size=reps)
    return np.concatenate([gammas, betas])


def optimize_single_start(sim: FastQAOASimulator, x0: Sequence[float],
                          start_index: int, maxiter: int = DEFAULT_MAXITER,
                          rhobeg: float = DEFAULT_RHOBEG) -> StartResult:
    """Run one COBYLA optimization and record everything about it."""
    import time
    from scipy.optimize import minimize

    counter = {"nfev": 0}

    def objective(theta: np.ndarray) -> float:
        counter["nfev"] += 1
        val = sim.energy(theta)
        if not np.isfinite(val):
            return 1e6
        return val

    t0 = time.perf_counter()
    res = minimize(objective, np.asarray(x0, dtype=float), method="COBYLA",
                   options={"maxiter": int(maxiter), "rhobeg": float(rhobeg)})
    elapsed = time.perf_counter() - t0

    x_final = np.asarray(res.x, dtype=float)
    valid = bool(np.all(np.isfinite(x_final)) and np.isfinite(res.fun))
    return StartResult(
        start_index=int(start_index),
        x0=[float(v) for v in np.asarray(x0, dtype=float)],
        x_final=[float(v) for v in x_final],
        energy_final=float(res.fun) if valid else float("nan"),
        nfev=int(counter["nfev"]),
        success=bool(res.success) and bool(res.status == 0) and valid,
        status=int(res.status),
        message=str(res.message),
        elapsed_s=float(elapsed),
    )


def multistart_optimize(sim: FastQAOASimulator, n_starts: int,
                        seed: int, maxiter: int = DEFAULT_MAXITER,
                        rhobeg: float = DEFAULT_RHOBEG
                        ) -> Tuple[List[StartResult], StartResult]:
    """Run ``n_starts`` distinct-seeded COBYLA starts; return (all, best_valid)."""
    rng = np.random.default_rng(seed)
    starts: List[StartResult] = []
    for i in range(n_starts):
        x0 = random_initial_parameters(rng, sim.reps)
        starts.append(optimize_single_start(sim, x0, i,
                                            maxiter=maxiter, rhobeg=rhobeg))
    valid = [s for s in starts if np.isfinite(s.energy_final)]
    if not valid:
        raise RuntimeError("no valid (finite-energy) optimization start")
    best = min(valid, key=lambda s: s.energy_final)
    return starts, best


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def exact_probabilities(qc: QAOACircuit, theta: Sequence[float]) -> np.ndarray:
    """Exact computational-basis probability distribution of the QAOA state."""
    vec = qaoa_statevector(qc, theta).data
    probs = np.abs(vec) ** 2
    total = probs.sum()
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"probabilities do not sum to 1 (got {total})")
    return probs / total


def sample_counts_exact(probs: np.ndarray, shots: int, seed: int) -> Dict[int, int]:
    """Multinomial-sample ``shots`` bitstring indices from exact probabilities.

    These are NOT hardware measurements: they are draws from the exact
    statevector probability distribution computed by Qiskit.
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    rng = np.random.default_rng(seed)
    draws = rng.choice(probs.shape[0], size=shots, p=probs)
    values, counts = np.unique(draws, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class MetricContext:
    """Per-instance constants needed to score a distribution."""

    n: int
    K: int
    mu: np.ndarray
    Sigma: np.ndarray
    q: float
    lam: float
    C_best: float
    C_worst: float
    optimal_indices: frozenset = field(default_factory=frozenset)

    @staticmethod
    def build(inst: core.PortfolioInstance, lam: float,
              benchmark: Dict) -> "MetricContext":
        optimal = set()
        for x in benchmark["all_optimal_assignments"]:
            optimal.add(core.assets_to_index(x))
        return MetricContext(
            n=inst.n, K=inst.K, mu=np.asarray(inst.mu, float),
            Sigma=np.asarray(inst.Sigma, float), q=inst.q, lam=lam,
            C_best=float(benchmark["best_feasible_objective"]),
            C_worst=float(benchmark["worst_feasible_objective"]),
            optimal_indices=frozenset(optimal),
        )


def _score_from_means(mean_c0_feasible: float, C_best: float,
                      C_worst: float) -> Optional[float]:
    denom = C_worst - C_best
    if abs(denom) <= 1e-15:
        return None
    return float((C_worst - mean_c0_feasible) / denom)


def aggregate_metrics(weights: Dict[int, float], total: float,
                      ctx: MetricContext) -> Dict[str, Optional[float]]:
    """Raw + normalised metrics from a weight distribution over basis indices.

    ``weights`` maps state index -> weight (probability mass or shot count);
    ``total`` is the sum of all weights (1.0 for probabilities, ``shots`` for
    counts). Infeasible states never enter the conditional feasible score.
    """
    if total <= 0:
        raise ValueError("total weight must be positive")
    feas_mass = 0.0
    feas_c0_sum = 0.0
    pen_sum = 0.0
    opt_mass = 0.0
    best_feas_c0 = math.inf
    for idx, w in weights.items():
        x = core.index_to_assets(int(idx), ctx.n)
        c0 = core.cost_unpenalized(x, ctx.mu, ctx.Sigma, ctx.q)
        pen = c0 + ctx.lam * (float(x.sum()) - ctx.K) ** 2
        pen_sum += w * pen
        if int(x.sum()) == ctx.K:
            feas_mass += w
            feas_c0_sum += w * c0
            if c0 < best_feas_c0:
                best_feas_c0 = c0
            if int(idx) in ctx.optimal_indices:
                opt_mass += w

    feas_rate = feas_mass / total
    score: Optional[float] = None
    mean_c0_feasible: Optional[float] = None
    if feas_mass > 0:
        mean_c0_feasible = feas_c0_sum / feas_mass
        score = _score_from_means(mean_c0_feasible, ctx.C_best, ctx.C_worst)
    return {
        "expected_penalized_cost": pen_sum / total,
        "feasibility_rate": feas_rate,
        "optimal_solution_probability": opt_mass / total,
        "best_feasible_unpenalized_cost": (float(best_feas_c0)
                                           if math.isfinite(best_feas_c0) else None),
        "mean_c0_conditioned_on_feasible": mean_c0_feasible,
        "feasible_normalized_score": score,
        "C_best": ctx.C_best,
        "C_worst": ctx.C_worst,
        "feasible_mass": feas_mass,
        "total_mass": total,
    }


def metrics_from_probs(probs: np.ndarray, ctx: MetricContext
                       ) -> Dict[str, Optional[float]]:
    """Exact statevector metrics from the full probability vector."""
    idx = np.nonzero(probs > 0)[0]
    weights = {int(i): float(probs[i]) for i in idx}
    return aggregate_metrics(weights, float(probs.sum()), ctx)


def metrics_from_counts(counts: Dict[int, int], ctx: MetricContext
                        ) -> Dict[str, Optional[float]]:
    """Sampled metrics from a shot-count dictionary."""
    return aggregate_metrics({int(k): float(v) for k, v in counts.items()},
                             float(sum(counts.values())), ctx)


# ---------------------------------------------------------------------------
# Exact penalty analysis (independent of any QAOA result)
# ---------------------------------------------------------------------------
def exact_penalty_analysis(inst: core.PortfolioInstance,
                           lambdas: Sequence[float], tol: float = 1e-12
                           ) -> Dict:
    """Exhaustive 2^n analysis of the penalized problem for each lambda.

    For every lambda, reports whether the global penalized minimizer is
    feasible, whether the feasible optimum is a global minimum, whether any
    infeasible assignment beats it, and the gap. Also derives the smallest
    penalty threshold that forces the feasible optimum to be a global minimum,

        lambda* = max_{x infeasible, C0(x) < C0*} (C0* - C0(x)) / (|x| - K)^2,

    computed directly from the enumerated assignments (never from QAOA samples).
    """
    n, K, q = inst.n, inst.K, inst.q
    X = core.all_assignments(n)
    Xf = X.astype(float)
    C0 = q * np.einsum("ki,ij,kj->k", Xf, np.asarray(inst.Sigma, float), Xf) \
        - Xf @ np.asarray(inst.mu, float)
    counts = X.sum(axis=1).astype(int)
    feasible = counts == K
    C_best = float(np.min(C0[feasible]))

    # Exact penalty threshold.
    ratios = []
    for k in range(2 ** n):
        if feasible[k]:
            continue
        gap = C_best - float(C0[k])
        if gap > 0:
            ratios.append(gap / float((counts[k] - K) ** 2))
    threshold = max(ratios) if ratios else 0.0

    per_lambda = {}
    for lam in lambdas:
        pen = C0 + lam * (counts - K) ** 2
        gmin = float(np.min(pen))
        argmin_idx = int(np.argmin(pen))
        best_feas_pen = float(np.min(pen[feasible]))
        gap = gmin - best_feas_pen
        per_lambda[f"{lam:g}"] = {
            "lambda": float(lam),
            "global_penalized_min": gmin,
            "global_minimizer_index": argmin_idx,
            "global_minimizer_assignment": X[argmin_idx].astype(int).tolist(),
            "global_minimizer_feasible": bool(feasible[argmin_idx]),
            "best_feasible_penalized_cost": best_feas_pen,
            "feasible_optimum_is_global_min": bool(gap >= -tol),
            "infeasible_beats_feasible": bool(gap < -tol),
            "gap_global_minus_feasible": float(gap),
        }
    return {
        "n": int(n),
        "K": int(K),
        "feasible_count": int(np.sum(feasible)),
        "best_feasible_unpenalized_cost": C_best,
        "penalty_threshold": float(threshold),
        "penalty_threshold_exists": bool(len(ratios) > 0),
        "per_lambda": per_lambda,
    }
