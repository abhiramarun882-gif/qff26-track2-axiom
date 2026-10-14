"""Phase 1 core: cardinality-constrained portfolio optimization model.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

Problem
-------
Minimize the synthetic portfolio cost

    C(x) = q * x^T Sigma x - mu^T x + lambda * (sum_i x_i - K)^2

over x in {0,1}^n. Lower is better. The feasible set is
F = { x in {0,1}^n : sum_i x_i = K }.

CONVENTIONS (used everywhere in this project)
---------------------------------------------
* Asset index ``i`` maps to qubit index ``i``.
* A computational-basis state is indexed by the integer
      s = sum_i x_i * 2^i,
  i.e. qubit 0 is the least-significant bit (Qiskit's little-endian order).
  ``index_to_assets(s, n)[i] = (s >> i) & 1``.
* In a Qiskit Pauli string the *rightmost* character is qubit 0, so a ``Z`` on
  qubit ``i`` sits at string position ``n - 1 - i``.
* Therefore the diagonal entry of the cost operator at matrix index ``s`` must
  equal ``cost_penalized(index_to_assets(s, n), ...)``.

This module is pure model/math. It builds no variational circuits and runs no
optimization, in line with the Phase 1 scope.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------------------------------------------------
# Documented numerical tolerances
# ---------------------------------------------------------------------------
OBJECTIVE_ATOL = 1e-9      # scalar objective agreement
MATRIX_ATOL = 1e-9         # operator <-> objective agreement
SYMMETRY_ATOL = 1e-12      # Sigma symmetry
PSD_ATOL = 1e-8            # positive-semidefiniteness of Sigma
HERMITICITY_ATOL = 1e-10   # mixer Hermiticity
COMMUTATOR_ATOL = 1e-10    # [H_M, sum_i Z_i] == 0
NORM_ATOL = 1e-10          # state normalisation
WEIGHT_ATOL = 1e-12        # Hamming-weight support checks

DEFAULT_SEED = 2026
DEFAULT_Q = 0.5


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioInstance:
    """A deterministic synthetic portfolio instance."""

    n: int
    K: int
    q: float
    seed: int
    mu: np.ndarray
    Sigma: np.ndarray

    def to_dict(self) -> Dict:
        """JSON-friendly representation (floats serialise via repr, round-trip safe)."""
        return {
            "n": int(self.n),
            "K": int(self.K),
            "q": float(self.q),
            "seed": int(self.seed),
            "mu": [float(v) for v in self.mu],
            "Sigma": [[float(v) for v in row] for row in self.Sigma],
        }


def _validate_generation_inputs(n: int, K: int, q: float) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not isinstance(K, (int, np.integer)) or isinstance(K, bool):
        raise TypeError(f"K must be an integer, got {type(K).__name__}")
    if K < 0 or K > n:
        raise ValueError(f"K must satisfy 0 <= K <= n, got K={K}, n={n}")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    if q < 0:
        raise ValueError(f"q must be non-negative, got {q}")


def generate_instance(n: int, K: int, q: float = DEFAULT_Q,
                      seed: int = DEFAULT_SEED) -> PortfolioInstance:
    """Generate a deterministic synthetic portfolio instance.

    Uses exactly ``np.random.default_rng(seed)`` with the draw order::

        mu    = rng.uniform(0.05, 0.25, size=n)
        A     = rng.normal(0.0, 0.1, size=(n, n))
        Sigma = A @ A.T

    ``Sigma`` is used as generated: no symmetrisation, normalisation or
    rescaling. Beginner (n=6, K=3) and Intermediate (n=12, K=6) instances are
    generated independently with the same seed and draw order for their own
    dimensions; neither is an extension of the other.
    """
    _validate_generation_inputs(n, K, q)
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.normal(0.0, 0.1, size=(n, n))
    Sigma = A @ A.T
    return PortfolioInstance(n=int(n), K=int(K), q=float(q), seed=int(seed),
                             mu=np.asarray(mu, dtype=float),
                             Sigma=np.asarray(Sigma, dtype=float))


def check_instance_properties(inst: PortfolioInstance) -> Dict[str, float]:
    """Return symmetry and PSD diagnostics for ``inst.Sigma``."""
    asym = float(np.max(np.abs(inst.Sigma - inst.Sigma.T))) if inst.n else 0.0
    eigvals = np.linalg.eigvalsh(inst.Sigma) if inst.n else np.array([0.0])
    return {
        "max_asymmetry": asym,
        "min_eigenvalue": float(np.min(eigvals)),
        "is_symmetric": bool(asym <= SYMMETRY_ATOL),
        "is_psd": bool(np.min(eigvals) >= -PSD_ATOL),
    }


# ---------------------------------------------------------------------------
# Classical objectives
# ---------------------------------------------------------------------------
def _validate_assignment(x: Sequence, n: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"x must be a 1-D vector of length {n}, got shape {arr.shape}")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("x must contain only binary values (0/1); "
                         "invalid vectors are rejected, not rounded")
    return arr.astype(float)


def cost_unpenalized(x: Sequence, mu: Sequence, Sigma: Sequence,
                     q: float = DEFAULT_Q) -> float:
    """C_0(x) = q x^T Sigma x - mu^T x."""
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    xa = _validate_assignment(x, mu.shape[0])
    return float(q * xa @ Sigma @ xa - mu @ xa)


def cost_penalized(x: Sequence, mu: Sequence, Sigma: Sequence, K: int,
                   lam: float, q: float = DEFAULT_Q) -> float:
    """C_lambda(x) = C_0(x) + lambda (sum_i x_i - K)^2."""
    if lam < 0 or not np.isfinite(lam):
        raise ValueError(f"lambda must be finite and non-negative, got {lam}")
    mu = np.asarray(mu, dtype=float)
    xa = _validate_assignment(x, mu.shape[0])
    penalty = lam * (float(np.sum(xa)) - K) ** 2
    return float(cost_unpenalized(xa, mu, Sigma, q) + penalty)


def cost_penalized_loop(x: Sequence, mu: Sequence, Sigma: Sequence, K: int,
                        lam: float, q: float = DEFAULT_Q) -> float:
    """Independent loop-based evaluation of C_lambda.

    Deliberately does not share the core cost-evaluation logic of
    :func:`cost_penalized`: the quadratic form is expanded term by term and the
    penalty is accumulated in a plain loop. Used as an independent reference.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    xa = _validate_assignment(x, n)
    quad = 0.0
    for i in range(n):
        for j in range(n):
            quad += q * xa[i] * Sigma[i, j] * xa[j]
    lin = 0.0
    for i in range(n):
        lin += mu[i] * xa[i]
    count = 0
    for i in range(n):
        count += xa[i]
    return float(quad - lin + lam * (count - K) ** 2)


def is_feasible(x: Sequence, K: int) -> bool:
    """True iff ``x`` is binary and has exactly ``K`` selected assets."""
    arr = np.asarray(x)
    if arr.ndim != 1 or not np.all((arr == 0) | (arr == 1)):
        return False
    return int(np.sum(arr)) == int(K)


# ---------------------------------------------------------------------------
# Bitstring / index conventions
# ---------------------------------------------------------------------------
def index_to_assets(index: int, n: int) -> np.ndarray:
    """x[i] = (index >> i) & 1 (qubit 0 = LSB, little-endian)."""
    if index < 0 or index >= 2 ** n:
        raise ValueError(f"index {index} out of range for n={n}")
    return np.array([(index >> i) & 1 for i in range(n)], dtype=int)


def assets_to_index(x: Sequence) -> int:
    """Inverse of :func:`index_to_assets`: index = sum_i x_i 2^i."""
    arr = np.asarray(x)
    if arr.ndim != 1 or not np.all((arr == 0) | (arr == 1)):
        raise ValueError("x must be a binary 1-D vector")
    return int(sum(int(v) << i for i, v in enumerate(arr)))


def all_assignments(n: int) -> np.ndarray:
    """Matrix of shape (2^n, n) with row ``s`` equal to ``index_to_assets(s, n)``."""
    return np.array([[ (s >> i) & 1 for i in range(n)] for s in range(2 ** n)],
                    dtype=int)


def feasible_assignments(n: int, K: int) -> np.ndarray:
    """Matrix of shape (C(n,K), n) enumerating all weight-K portfolios."""
    if K < 0 or K > n:
        raise ValueError(f"K must satisfy 0 <= K <= n, got K={K}, n={n}")
    rows = []
    for combo in itertools.combinations(range(n), K):
        x = np.zeros(n, dtype=int)
        x[list(combo)] = 1
        rows.append(x)
    return np.array(rows, dtype=int)


# ---------------------------------------------------------------------------
# QUBO representation
# ---------------------------------------------------------------------------
def qubo_coefficients(mu: Sequence, Sigma: Sequence, q: float, K: int,
                      lam: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (c, a, B) with C_lambda(x) = c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j.

    Conventions (exact):

        a_i    = q Sigma_ii - mu_i + lam (1 - 2K)
        b_ij   = 2 q Sigma_ij + 2 lam      (i < j)
        c      = lam K^2

    ``a`` has shape (n,); ``B`` has shape (n, n) with ``B[i, j] = b_ij`` for
    i < j and 0 elsewhere (strictly upper triangular). Diagonal contributions
    (q Sigma_ii x_i, using x_i^2 = x_i) are folded into ``a_i``; the constant
    lam K^2 and the penalty cross terms are folded in exactly once.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError(f"Sigma must be ({n},{n}), got {Sigma.shape}")
    if lam < 0 or not np.isfinite(lam):
        raise ValueError(f"lambda must be finite and non-negative, got {lam}")

    a = q * np.diag(Sigma) - mu + lam * (1 - 2 * K)
    B = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            B[i, j] = 2.0 * q * Sigma[i, j] + 2.0 * lam
    c = lam * K ** 2
    return float(c), a, B


def cost_from_qubo(x: Sequence, c: float, a: Sequence, B: Sequence) -> float:
    """Evaluate the QUBO form c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j."""
    a = np.asarray(a, dtype=float)
    B = np.asarray(B, dtype=float)
    n = a.shape[0]
    xa = _validate_assignment(x, n)
    val = float(c)
    val += float(a @ xa)
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] != 0.0:
                val += float(B[i, j]) * xa[i] * xa[j]
    return val


# ---------------------------------------------------------------------------
# QUBO -> Pauli operator
# ---------------------------------------------------------------------------
def _z_pauli_string(n: int, z_indices: Sequence[int]) -> str:
    """Pauli string with Z on ``z_indices``; rightmost char is qubit 0."""
    chars = ["I"] * n
    for i in z_indices:
        chars[n - 1 - i] = "Z"
    return "".join(chars)


def qubo_to_pauli(mu: Sequence, Sigma: Sequence, q: float, K: int, lam: float
                  ) -> Tuple[SparsePauliOp, float, np.ndarray, np.ndarray]:
    """Map the QUBO to H_C = c_I I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.

    With x_i = (I - Z_i)/2:

        c_I  = c + 1/2 sum_i a_i + 1/4 sum_{i<j} b_ij
        h_i  = -a_i/2 - 1/4 sum_{j != i} b_{min(i,j),max(i,j)}
        J_ij = b_ij / 4

    Returns ``(op, c_I, h, J)`` where ``J`` is upper triangular (J[i,j]=J_ij
    for i<j, 0 elsewhere). The identity coefficient ``c_I`` is preserved.
    """
    c, a, B = qubo_coefficients(mu, Sigma, q, K, lam)
    n = a.shape[0]

    pair_sum = float(np.sum(B))
    c_I = c + 0.5 * float(np.sum(a)) + 0.25 * pair_sum

    h = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            lo, hi = (i, j) if i < j else (j, i)
            s += B[lo, hi]
        h[i] = -0.5 * a[i] - 0.25 * s

    J = B / 4.0

    terms: List[Tuple[str, float]] = [("I" * n, float(c_I))]
    for i in range(n):
        if h[i] != 0.0:
            terms.append((_z_pauli_string(n, [i]), float(h[i])))
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                terms.append((_z_pauli_string(n, [i, j]), float(J[i, j])))

    op = SparsePauliOp.from_list(terms)
    return op, float(c_I), h, J


def operator_diagonal(op: SparsePauliOp) -> np.ndarray:
    """Real diagonal of a diagonal (I/Z/ZZ) Pauli operator, computed by Qiskit."""
    mat = op.to_matrix(sparse=True)
    diag = np.asarray(mat.diagonal()).ravel()
    if np.max(np.abs(diag.imag)) > 1e-12:
        raise ValueError("operator is not diagonal as expected")
    return diag.real.copy()


# ---------------------------------------------------------------------------
# XY ring mixer
# ---------------------------------------------------------------------------
def build_xy_ring_mixer(n: int, coupling: float = 0.5) -> SparsePauliOp:
    """Nearest-neighbour ring mixer

        H_M = (coupling) * sum_{i=0}^{n-1} (X_i X_{(i+1) mod n} + Y_i Y_{(i+1) mod n}),

    with ``coupling = 0.5`` matching the challenge's 1/2 prefactor and periodic
    (wrap-around) boundary conditions. This is an XY-type mixer, not an X-only
    mixer, and conserves total excitation number (Hamming weight).

    Special case n = 2: the periodic expression lists the single undirected
    edge (0, 1) twice (as i=0 and i=1), so the effective XX/YY coupling is
    doubled. This literal convention is preserved and documented rather than
    silently deduplicated.
    """
    if n < 2:
        raise ValueError(f"ring mixer requires n >= 2, got {n}")
    terms: List[Tuple[str, float]] = []
    for i in range(n):
        j = (i + 1) % n
        x_str = ["I"] * n
        x_str[n - 1 - i] = "X"
        x_str[n - 1 - j] = "X"
        terms.append(("".join(x_str), coupling))
        y_str = ["I"] * n
        y_str[n - 1 - i] = "Y"
        y_str[n - 1 - j] = "Y"
        terms.append(("".join(y_str), coupling))
    op = SparsePauliOp.from_list(terms)
    op = op.simplify()  # merge duplicate (0,1) terms for n=2, summing coefficients
    return op


def excitation_number_operator(n: int) -> SparsePauliOp:
    """sum_i (I - Z_i)/2, whose expectation is the number of selected assets."""
    terms = [("I" * n, n / 2.0)]
    for i in range(n):
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        terms.append(("".join(s), -0.5))
    return SparsePauliOp.from_list(terms)


def hamming_weight(index: int) -> int:
    return int(bin(index).count("1"))


# ---------------------------------------------------------------------------
# Fixed-weight (Dicke-like) initial state
# ---------------------------------------------------------------------------
def uniform_weight_state(n: int, K: int) -> np.ndarray:
    """Amplitude vector of the normalised uniform fixed-weight state

        |D^n_K> = 1/sqrt(C(n,K)) * sum_{|x|=K} |x>.

    The vector has dimension 2^n, is zero on infeasible basis states, and has
    exactly C(n,K) equal nonzero amplitudes. The index convention matches the
    rest of this module (qubit 0 = LSB).
    """
    if K < 0 or K > n:
        raise ValueError(f"K must satisfy 0 <= K <= n, got K={K}, n={n}")
    count = math.comb(n, K)
    vec = np.zeros(2 ** n, dtype=complex)
    amp = 1.0 / math.sqrt(count)
    for s in range(2 ** n):
        if bin(s).count("1") == K:
            vec[s] = amp
    return vec


def weight_state_support_counts(n: int, K: int) -> Dict[str, float]:
    """Diagnostics for the uniform weight-K state."""
    vec = uniform_weight_state(n, K)
    nonzero = int(np.count_nonzero(vec))
    norm = float(np.linalg.norm(vec))
    support = int(np.where(np.abs(vec) > 1e-15)[0].shape[0])
    support_ok = all(bin(s).count("1") == K for s in np.where(np.abs(vec) > 1e-15)[0])
    return {
        "dimension": int(vec.shape[0]),
        "expected_nonzero": int(math.comb(n, K)),
        "nonzero_amplitudes": nonzero,
        "support_count": support,
        "support_is_weight_K": bool(support_ok),
        "norm": norm,
        "norm_error": float(abs(norm - 1.0)),
    }


# ---------------------------------------------------------------------------
# Exact classical benchmark
# ---------------------------------------------------------------------------
def _objective_matrix(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                      q: float) -> np.ndarray:
    """Unpenalized C_0 for every row of X (vectorised)."""
    Xf = X.astype(float)
    return q * np.einsum("ki,ij,kj->k", Xf, Sigma, Xf) - Xf @ mu


def exact_benchmark(n: int, K: int, mu: Sequence, Sigma: Sequence, q: float,
                    lam_ref: float = 1.0, tol: float = OBJECTIVE_ATOL) -> Dict:
    """Exhaustive feasible benchmark plus independent cross-checks.

    Primary path: enumerate all C(n,K) weight-K portfolios with
    :func:`feasible_assignments` and evaluate the vectorised objective.
    Independent path: enumerate every basis state by bitmask and evaluate the
    loop-based cost, keeping the feasible ones. The two paths share no core
    cost logic.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if mu.shape[0] != n or Sigma.shape != (n, n):
        raise ValueError("shape mismatch between n, mu and Sigma")

    feasible = feasible_assignments(n, K)
    feasible_count = int(feasible.shape[0])
    expected_count = math.comb(n, K)

    obj = _objective_matrix(feasible, mu, Sigma, q)
    best_val = float(np.min(obj))
    worst_val = float(np.max(obj))
    best_mask = np.isclose(obj, best_val, atol=tol, rtol=0.0)
    best_rows = np.where(best_mask)[0]
    # deterministic tie handling: report lexicographically smallest index
    best_local = int(best_rows[0])
    best_indices = tuple(sorted(int(i) for i in np.where(feasible[best_local])[0]))
    best_assignment = feasible[best_local].astype(int).tolist()

    order = np.argsort(obj, kind="stable")
    sorted_objectives = [float(obj[i]) for i in order]

    # Independent bitmask enumeration (loop-based cost).
    indep_best_val = math.inf
    indep_best_rows: List[int] = []
    indep_count = 0
    for s in range(2 ** n):
        x = index_to_assets(s, n)
        if int(x.sum()) != K:
            continue
        indep_count += 1
        val = cost_penalized_loop(x, mu, Sigma, K, 0.0, q)  # feasible: penalty term is 0
        if val < indep_best_val - tol:
            indep_best_val = val
            indep_best_rows = [s]
        elif abs(val - indep_best_val) <= tol:
            indep_best_rows.append(s)
    indep_best_local = min(indep_best_rows)
    indep_indices = tuple(int(i) for i in np.where(index_to_assets(indep_best_local, n))[0])

    # Full-space penalized min/max at the documented reference penalty.
    allX = all_assignments(n)
    all_obj0 = _objective_matrix(allX, mu, Sigma, q)
    counts = allX.sum(axis=1).astype(float)
    all_pen = all_obj0 + lam_ref * (counts - K) ** 2

    # Independent verification of the optimum (both paths agree).
    independent_agreement = bool(
        indep_count == feasible_count
        and abs(float(indep_best_val) - best_val) <= tol
        and indep_indices == best_indices
    )

    return {
        "n": int(n),
        "K": int(K),
        "q": float(q),
        "feasible_count": feasible_count,
        "expected_feasible_count": int(expected_count),
        "feasible_count_ok": bool(feasible_count == expected_count),
        "optimal_unpenalized_cost": best_val,
        "optimal_asset_indices": list(best_indices),
        "optimal_assignment": best_assignment,
        "num_optimal_ties": int(best_rows.shape[0]),
        "all_optimal_assignments": feasible[best_rows].astype(int).tolist(),
        "best_feasible_objective": best_val,
        "worst_feasible_objective": worst_val,
        "sorted_feasible_objectives": sorted_objectives,
        "tie_tolerance": float(tol),
        "lam_ref": float(lam_ref),
        "full_space_penalized_min": float(np.min(all_pen)),
        "full_space_penalized_max": float(np.max(all_pen)),
        "independent_feasible_count": int(indep_count),
        "independent_optimal_cost": float(indep_best_val),
        "independent_optimal_indices": list(indep_indices),
        "independent_agreement": independent_agreement,
    }
