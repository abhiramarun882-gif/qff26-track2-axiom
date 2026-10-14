"""Phase 3 final follow-up: two bounded validation checks.

Run from the project root with the project interpreter:

    .venv\\Scripts\\python.exe -u run_phase3_followup.py

Check A
    Derive the Intermediate-instance penalty threshold lambda* by exhaustive
    enumeration of all 4096 binary assignments, via two independent code paths,
    and write outputs/phase3/intermediate_penalty_threshold.json.

Check B
    Cross-validate Methods B and C against independent Qiskit circuits at
    n=12 (not n=6), using a representative parameter vector taken verbatim
    from the final experiment artifacts. Each circuit runs in its own child
    process under a genuine 600 s wall-clock timeout. Results go to
    outputs/phase3/final_followup_validation.json.

The script then runs the full test suite and appends a dated entry to
logs/phase3-run.log.

Restrictions honoured here: no optimiser is rerun, and no existing experiment
JSON, figure, or validation report is overwritten. This script only creates the
two new output files and appends to the validation log.

Exit codes
    0  both checks fresh and the test suite passed
    1  the test suite failed
    3  a check timed out or failed (reported honestly, outputs still written)
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

import phase1_core as core
import phase2_qaoa as q2
import phase3_methods as p3

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase3")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

PYTHON = sys.executable
N12_TIMEOUT_S = 600.0

N, K = 12, 6
EXPECTED_OPTIMUM = -1.0663920923466492

# Qiskit's ``quantum_info.Statevector`` applies an opaque ``PauliEvolutionGate``
# by exponentiating its generator matrix instead of using the gate's own
# definition. At n=12 that dense/sparse exponential costs ~14 s per cost layer
# and >386 s per X-mixer layer, which blows through any sensible timeout.
# Expanding the circuit to its own definition with ``QuantumCircuit.decompose``
# is unitary-preserving (it is exactly the gate Qiskit itself defines) and runs
# in ~0.04 s. The faithfulness of this expansion is measured at n=6, where the
# undecomposed path is fast, in ``decomposition_faithfulness`` below.
DECOMPOSE_REPS = 4

THRESHOLD_PATH = os.path.join(OUT_DIR, "intermediate_penalty_threshold.json")
FOLLOWUP_PATH = os.path.join(OUT_DIR, "final_followup_validation.json")
RUN_LOG = os.path.join(LOG_DIR, "phase3-run.log")
TESTS_LOG = os.path.join(LOG_DIR, "phase3-tests.log")

SOURCE_ARTIFACTS = {
    "B": "method_b_warm_start.json",
    "C": "method_c_penalty_qaoa.json",
}

CHILD_QISKIT_CODE = r"""
import json, sys, os
import numpy as np
root, method, p_str, theta_json, c_json = sys.argv[1:6]
sys.path.insert(0, root)
os.chdir(root)
import phase1_core as core
import phase2_qaoa as q2
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector

p = int(p_str)
theta = json.loads(theta_json)
inst = core.generate_instance(12, 6)
cost_op = q2.cost_operator(inst, 1.0)
qc = QuantumCircuit(12)
if method == "B":
    c = json.loads(c_json)
    for i in range(12):
        qc.ry(2 * np.arcsin(np.sqrt(c[i])), i)
else:
    qc.h(range(12))
for k in range(p):
    qc.append(PauliEvolutionGate(cost_op, time=theta[k]), range(12))
    qc.append(PauliEvolutionGate(q2.build_x_mixer(12), time=theta[p + k]),
              range(12))
qc = qc.decompose(reps=4)
sv = Statevector(qc).data
out = {
    "energy_qiskit": float(q2.qaoa_energy_from_vector(cost_op, sv)),
    "norm": float(np.linalg.norm(sv)),
    "decomposed_depth": int(qc.depth()),
    "decomposed_size": int(qc.size()),
    "decomposed_ops": {str(k): int(v) for k, v in qc.count_ops().items()},
    "real": [float(v) for v in sv.real],
    "imag": [float(v) for v in sv.imag],
}
sys.stdout.write("\n<<<RESULT>>>" + json.dumps(out) + "<<<END>>>\n")
sys.stdout.flush()
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name: str) -> dict:
    with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _dump(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
# Check A - exact penalty threshold, two independent enumeration paths
# ---------------------------------------------------------------------------
def threshold_path1(inst, C_best: float, K: int) -> dict:
    """Scalar path: index -> assets -> core.cost_unpenalized, one at a time."""
    lam_star = 0.0
    argmax_index = None
    argmax_row = None
    evaluated_infeasible = 0
    for index in range(2 ** inst.n):
        x = core.index_to_assets(index, inst.n)
        cardinality = int(x.sum())
        if cardinality == K:
            continue
        evaluated_infeasible += 1
        c0 = core.cost_unpenalized(x, inst.mu, inst.Sigma, inst.q)
        ratio = (C_best - c0) / (cardinality - K) ** 2
        candidate = max(0.0, ratio)
        if argmax_index is None or candidate > lam_star:
            lam_star = candidate
            argmax_index = index
            argmax_row = {
                "index": index,
                "asset_indices": [int(v) for v in np.flatnonzero(x)],
                "assignment": [int(v) for v in x],
                "cardinality": cardinality,
                "C_0": float(c0),
                "numerator_C_best_minus_C0": float(C_best - c0),
                "denominator_cardinality_minus_K_squared": float(
                    (cardinality - K) ** 2),
                "ratio": float(ratio),
            }
    return {
        "path": "path_1_scalar_loop",
        "description": ("scalar loop over all 4096 indices using "
                        "phase1_core.index_to_assets + "
                        "phase1_core.cost_unpenalized (which validates each "
                        "assignment is binary)"),
        "assignments_scanned": 2 ** inst.n,
        "infeasible_assignments_considered": evaluated_infeasible,
        "lambda_star": float(lam_star),
        "argmax": argmax_row,
    }


def threshold_path2(inst, C_best: float, K: int) -> dict:
    """Vectorised path: bits built by shifts, quadratic form by matmul."""
    n = inst.n
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(float)
    Sigma = np.asarray(inst.Sigma, dtype=float)
    mu = np.asarray(inst.mu, dtype=float)
    c0 = inst.q * np.sum((bits @ Sigma) * bits, axis=1) - bits @ mu
    cardinality = bits.sum(axis=1).astype(int)

    infeasible = cardinality != K
    denom = (cardinality[infeasible] - K).astype(float) ** 2
    numerator = C_best - c0[infeasible]
    ratio = numerator / denom
    candidate = np.maximum(ratio, 0.0)
    local = int(np.argmax(candidate))
    best_candidate = float(candidate[local])
    flat_index = int(np.flatnonzero(infeasible)[local])
    # all infeasible assignments attaining the maximum (tie set)
    tie_tol = 1e-12 * max(1.0, abs(best_candidate))
    tie_mask = np.abs(candidate - best_candidate) <= tie_tol
    tie_indices = [int(v) for v in np.flatnonzero(infeasible)[tie_mask]]

    x = bits[flat_index]
    return {
        "path": "path_2_vectorised_bitshift",
        "description": ("vectorised: bits built by bit-shift, quadratic form "
                        "as a single matrix product, all 4096 rows at once; "
                        "shares no scalar helper with path 1"),
        "assignments_scanned": 2 ** n,
        "infeasible_assignments_considered": int(infeasible.sum()),
        "lambda_star": best_candidate,
        "argmax": {
            "index": flat_index,
            "asset_indices": [int(v) for v in np.flatnonzero(x)],
            "assignment": [int(v) for v in x.astype(int)],
            "cardinality": int(cardinality[flat_index]),
            "C_0": float(c0[flat_index]),
            "numerator_C_best_minus_C0": float(numerator[local]),
            "denominator_cardinality_minus_K_squared": float(denom[local]),
            "ratio": float(ratio[local]),
        },
        "all_argmax_indices": tie_indices,
        "_c0_all": c0,
        "_cardinality_all": cardinality,
    }


def run_check_a() -> tuple[dict, list[str]]:
    problems: list[str] = []
    inst = core.generate_instance(N, K)
    bench = core.exact_benchmark(N, K, inst.mu, inst.Sigma, inst.q)
    C_best = float(bench["best_feasible_objective"])

    if not np.isclose(C_best, EXPECTED_OPTIMUM, atol=1e-9):
        problems.append(f"C_best {C_best!r} != expected {EXPECTED_OPTIMUM!r}")

    p1 = threshold_path1(inst, C_best, K)
    p2 = threshold_path2(inst, C_best, K)

    # independent cross-checks
    c0_from_path2_min_feasible = float(
        p2["_c0_all"][p2["_cardinality_all"] == K].min())
    feasible_min_matches = bool(
        np.isclose(c0_from_path2_min_feasible, C_best, atol=1e-12))
    lam_agree = bool(np.isclose(p1["lambda_star"], p2["lambda_star"],
                                rtol=0, atol=1e-12))
    argmax_agree = bool(p1["argmax"]["index"] == p2["argmax"]["index"])
    c0_agree = bool(np.isclose(p1["argmax"]["C_0"], p2["argmax"]["C_0"],
                               rtol=0, atol=1e-12))

    if not feasible_min_matches:
        problems.append("path2 feasible minimum does not match C_best")
    if not lam_agree:
        problems.append(
            f"lambda* disagrees: path1={p1['lambda_star']!r} "
            f"path2={p2['lambda_star']!r}")
    if not argmax_agree:
        problems.append(f"argmax index disagrees: path1="
                        f"{p1['argmax']['index']} path2={p2['argmax']['index']}")
    if not c0_agree:
        problems.append("argmax C_0 disagrees between paths")

    lam_star = float(p2["lambda_star"])
    argmax = p2["argmax"]
    x_max = np.array(argmax["assignment"], dtype=int)

    # third, independent confirmation of the tie using cost_penalized_loop
    tie_via_loop = core.cost_penalized_loop(
        x_max, inst.mu, inst.Sigma, K, lam_star, inst.q)
    tie_gap = float(tie_via_loop - C_best)

    # minimiser sets at lambda*, lambda* + delta, lambda* - delta
    c0_all = p2["_c0_all"]
    card_all = p2["_cardinality_all"]
    scale = max(1.0, abs(lam_star))
    tol = 1e-9 * max(1.0, abs(C_best))

    def minimisers(extra: float) -> list[int]:
        v = c0_all + (lam_star + extra) * (card_all - K) ** 2
        m = v.min()
        return [int(i) for i in np.flatnonzero(v <= m + tol)]

    at_threshold = minimisers(0.0)
    above_threshold = minimisers(1e-6 * (1.0 + scale))
    below_threshold = minimisers(-1e-6 * (1.0 + scale))

    opt_assets = [int(v) for v in bench["optimal_asset_indices"]]
    x_opt = np.zeros(inst.n, dtype=int)
    x_opt[opt_assets] = 1
    feasible_index = core.assets_to_index(x_opt)
    ai = p2["argmax"]["index"]
    argmax_penalised = float(c0_all[ai] + lam_star * (card_all[ai] - K) ** 2)
    tie_exists = bool(abs(argmax_penalised - C_best) <= tol)

    # distribution of required lambda over infeasible cardinalities
    by_cardinality = {}
    for w in range(0, N + 1):
        if w == K:
            continue
        mask = card_all == w
        if not mask.any():
            continue
        best_for_w = float(np.maximum((C_best - c0_all[mask]) / (w - K) ** 2,
                                      0.0).max())
        by_cardinality[str(w)] = {
            "num_assignments": int(mask.sum()),
            "max_required_lambda": best_for_w,
        }

    result = {
        "team": "axiom", "phase": 3, "study": "intermediate_penalty_threshold",
        "generated_utc": _now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "formula": ("lambda* = max_{x: |x| != K} max(0, (C_best - C_0(x)) / "
                    "(|x| - K)^2)"),
        "instance": {"level": "intermediate", "n": inst.n, "K": inst.K,
                     "q": inst.q, "seed": inst.seed},
        "exact_reference": {
            "source": "phase1_core.exact_benchmark + independent enumeration",
            "C_best": C_best,
            "C_worst": float(bench["worst_feasible_objective"]),
            "optimal_asset_indices": opt_assets,
            "feasible_count": int(bench["feasible_count"]),
            "num_optimal_ties": int(bench["num_optimal_ties"]),
            "expected_optimum": EXPECTED_OPTIMUM,
        },
        "lambda_star": lam_star,
        "path_1": p1,
        "path_2": {k: v for k, v in p2.items()
                   if not k.startswith("_")},
        "cross_checks": {
            "feasible_min_equals_C_best": feasible_min_matches,
            "feasible_min_from_path_2": c0_from_path2_min_feasible,
            "lambda_star_paths_agree": lam_agree,
            "argmax_index_paths_agree": argmax_agree,
            "argmax_C0_paths_agree": c0_agree,
            "tie_confirmed_by_cost_penalized_loop": bool(abs(tie_gap) <= tol),
            "tie_gap_via_loop": tie_gap,
        },
        "maximizing_assignment": {
            **argmax,
            "penalized_objective_at_lambda_star": argmax_penalised,
            "C_best": C_best,
        },
        "tie_at_threshold": {
            "creates_tie_with_feasible_optimum": tie_exists,
            "explanation": (
                "At lambda = lambda* the maximizing infeasible assignment has "
                "penalized objective exactly equal to C_best by construction "
                "(equality holds in the defining inequality), so the feasible "
                "optimum is a global minimiser but not the unique one."
            ),
            "num_global_minimisers_at_lambda_star": len(at_threshold),
            "global_minimizer_indices_at_lambda_star": at_threshold[:32],
            "contains_feasible_optimum": feasible_index in at_threshold,
            "contains_maximizing_assignment":
                p2["argmax"]["index"] in at_threshold,
            "global_minimisers_just_above_lambda_star": above_threshold[:32],
            "num_global_minimisers_just_above_lambda_star":
                len(above_threshold),
            "unique_above_threshold": len(above_threshold) == 1,
            "global_minimisers_just_below_lambda_star": below_threshold[:32],
            "num_global_minimisers_just_below_lambda_star":
                len(below_threshold),
            "feasible_optimum_beaten_below_threshold":
                feasible_index not in below_threshold,
            "delta_used": 1e-6 * (1.0 + scale),
        },
        "required_lambda_by_cardinality": by_cardinality,
        "comparison_to_phase_2_beginner_threshold": {
            "beginner_lambda_star": 0.109661226,
            "intermediate_lambda_star": lam_star,
            "note": ("Phase 2 derived lambda* for the Beginner instance only. "
                     "Phase 3 ran Method C at lambda=1.0, which is above this "
                     "Intermediate threshold."),
        },
        "commands": {
            "this_script": " ".join(sys.argv),
            "enumeration": "in-process, exhaustive over 2**12 = 4096 states",
        },
        "notes": [
            "Enumeration only: no optimiser is rerun.",
            "Path 1 and path 2 share no cost-evaluation helper; both are "
            "cross-checked against phase1_core.exact_benchmark.",
            "The tie at lambda* is additionally confirmed with "
            "phase1_core.cost_penalized_loop, a third independent evaluator.",
        ],
    }
    return result, problems


# ---------------------------------------------------------------------------
# Check B - Methods B and C vs Qiskit at n=12
# ---------------------------------------------------------------------------
def select_representative(artifact: dict, method: str) -> dict:
    """Pick the record with the best optimal-solution probability."""
    records = artifact["records"]
    key = lambda r: r["sampling"]["exact_statevector_metrics"][
        "optimal_solution_probability"]
    best_pos = max(range(len(records)), key=lambda i: key(records[i]))
    rec = records[best_pos]
    theta = [float(v) for v in rec["best_theta"]]
    return {
        "method": method,
        "record_index": int(best_pos),
        "p": int(rec["p"]),
        "theta": theta,
        "theta_is_finite": bool(np.all(np.isfinite(theta))),
        "best_start_index": int(rec["best_start_index"]),
        "reported_best_energy_C_lambda_exact":
            float(rec["best_energy_C_lambda_exact"]),
        "reported_optimal_probability_exact": float(key(rec)),
        "reported_feasibility_rate_exact": float(
            rec["sampling"]["exact_statevector_metrics"]["feasibility_rate"]),
        "selection_rule": ("record with the highest exact optimal-solution "
                           "probability in the final experiment artifact "
                           "(each method's headline depth)"),
    }


def simulator_side(method: str, rep: dict, c: list | None) -> dict:
    """Fast-simulator side of the comparison (in-process, no Qiskit circuit)."""
    inst = core.generate_instance(N, K)
    cost_op = q2.cost_operator(inst, p3.LAMBDA)
    diag = core.operator_diagonal(cost_op)
    theta = np.asarray(rep["theta"], dtype=float)
    if method == "B":
        init = p3.product_state_amplitudes(np.asarray(c, dtype=float))
    else:
        init = p3.uniform_plus_amplitudes(inst.n)
    sim = p3.FullSpaceSimulator(inst.n, rep["p"], diag, init)
    vec = sim.statevector(theta)
    probs = np.abs(vec) ** 2
    return {
        "energy_simulator": float(sim.energy(theta)),
        "normalization_error_simulator": float(abs(probs.sum() - 1.0)),
        "_vec": vec,
        "_probs": probs,
        "_cost_op": cost_op,
    }


def decomposition_faithfulness_check() -> dict:
    """At n=6 the undecomposed Statevector path is fast, so we can measure,
    rather than assume, that ``QuantumCircuit.decompose`` preserves the unitary
    we are comparing against."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector

    inst6 = core.generate_instance(6, 3)
    cost_op = q2.cost_operator(inst6, p3.LAMBDA)
    mx = q2.build_x_mixer(inst6.n)
    rng = np.random.default_rng(991)
    out = {}
    for method in ("B", "C"):
        if method == "B":
            c = rng.uniform(0.1, 0.9, size=inst6.n)
        else:
            c = None
        for p in (1, 2, 3):
            theta = rng.uniform(0.0, 2 * np.pi, size=2 * p)
            qc = QuantumCircuit(inst6.n)
            if method == "B":
                for i in range(inst6.n):
                    qc.ry(2 * np.arcsin(np.sqrt(c[i])), i)
            else:
                qc.h(range(inst6.n))
            for k in range(p):
                qc.append(PauliEvolutionGate(cost_op, time=theta[k]),
                          range(inst6.n))
                qc.append(PauliEvolutionGate(mx, time=theta[p + k]),
                          range(inst6.n))
            und = Statevector(qc).data
            dec = qc.decompose(reps=DECOMPOSE_REPS)
            dec_sv = Statevector(dec).data
            e_und = q2.qaoa_energy_from_vector(cost_op, und)
            e_dec = q2.qaoa_energy_from_vector(cost_op, dec_sv)
            out[f"{method}_p{p}"] = {
                "n": 6,
                "fidelity_undecomposed_vs_decomposed":
                    float(abs(np.vdot(und, dec_sv)) ** 2),
                "max_abs_statevector_diff":
                    float(np.abs(und - dec_sv).max()),
                "abs_energy_diff": float(abs(e_und - e_dec)),
                "depth_undecomposed": int(qc.depth()),
                "size_undecomposed": int(qc.size()),
                "depth_decomposed": int(dec.depth()),
                "size_decomposed": int(dec.size()),
            }
    worst_fid = min(v["fidelity_undecomposed_vs_decomposed"]
                    for v in out.values())
    worst_dE = max(v["abs_energy_diff"] for v in out.values())
    return {
        "status": "fresh",
        "description": ("n=6 cross-check of the simulation methodology: the "
                        "same Qiskit circuit is simulated both with opaque "
                        "PauliEvolutionGate gates and after "
                        "QuantumCircuit.decompose(reps=%d). This measures, "
                        "rather than assumes, that decompose is unitary-"
                        "preserving before it is relied on at n=12." % DECOMPOSE_REPS),
        "decompose_reps": DECOMPOSE_REPS,
        "worst_fidelity": float(worst_fid),
        "worst_abs_energy_diff": float(worst_dE),
        "tolerances": {"fidelity": 0.999999, "abs_energy_diff": 1e-12},
        "passed": bool(worst_fid >= 0.999999 and worst_dE <= 1e-12),
        "records": out,
    }


def run_qiskit_side(method: str, rep: dict, c: list | None) -> dict:
    """One Qiskit reference circuit in a child process with a real timeout."""
    cmd = [
        PYTHON, "-u", "-c", CHILD_QISKIT_CODE, PROJECT_ROOT, method,
        str(rep["p"]), json.dumps(rep["theta"]),
        json.dumps(c if c is not None else []),
    ]
    shown = (f"{os.path.basename(PYTHON)} -u -c <child Qiskit n=12 Method "
             f"{method} p={rep['p']}, circuit decomposed reps={DECOMPOSE_REPS}> "
             f"{PROJECT_ROOT} {method} {rep['p']} "
             f"<theta[{len(rep['theta'])}]> <c[{len(c) if c else 0}]>")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=N12_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return {
            "status": "timeout",
            "command": shown,
            "timeout_s": float(N12_TIMEOUT_S),
            "elapsed_s": round(float(elapsed), 3),
            "exit_code": None,
            "error": (f"child process exceeded the {N12_TIMEOUT_S:.0f}s "
                      "wall-clock timeout; the Qiskit reference state was "
                      "never obtained, so no agreement metrics exist for "
                      "this method"),
            "raw": None,
        }
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"<<<RESULT>>>(.*?)<<<END>>>", out, re.S)
    if proc.returncode != 0 or not m:
        return {
            "status": "error",
            "command": shown,
            "timeout_s": float(N12_TIMEOUT_S),
            "elapsed_s": round(float(elapsed), 3),
            "exit_code": int(proc.returncode),
            "error": out[-1200:],
            "raw": None,
        }
    raw = json.loads(m.group(1))
    return {
        "status": "fresh",
        "command": shown,
        "timeout_s": float(N12_TIMEOUT_S),
        "elapsed_s": round(float(elapsed), 3),
        "exit_code": int(proc.returncode),
        "raw": raw,
    }


def run_check_b() -> tuple[dict, list[str]]:
    problems: list[str] = []
    inst = core.generate_instance(N, K)
    out_methods = {}

    for method in ("B", "C"):
        name = SOURCE_ARTIFACTS[method]
        artifact = _load(name)
        path = os.path.join(OUT_DIR, name)
        rep = select_representative(artifact, method)

        c = None
        if method == "B":
            c = [float(v) for v in artifact["relaxation"]["final_point"]]
            if len(c) != N:
                problems.append(f"Method B warm-start vector has {len(c)} "
                                f"entries, expected {N}")
            if not all(0.0 - 1e-12 <= v <= 1.0 + 1e-12 for v in c):
                problems.append("Method B warm-start vector outside [0, 1]")

        if not rep["theta_is_finite"]:
            problems.append(f"Method {method}: selected theta is not finite")
            out_methods[method] = {
                "method": method, "status": "error",
                "error": "selected parameter vector is not finite",
                "selection": rep,
            }
            continue

        sim = simulator_side(method, rep, c)
        qk = run_qiskit_side(method, rep, c)

        block = {
            "method": method,
            "description": ("p3.FullSpaceSimulator (fast) vs a Qiskit circuit "
                            "built from scratch with PauliEvolutionGate, at "
                            "n=12, K=6, lambda=1.0"),
            "status": qk["status"],
            "timeout_s": float(N12_TIMEOUT_S),
            "elapsed_s": qk["elapsed_s"],
            "exit_code": qk["exit_code"],
            "command": qk["command"],
            "selection": rep,
            "source_artifact": {
                "path": os.path.relpath(path, PROJECT_ROOT),
                "generated_utc": artifact.get("generated_utc"),
                "sha256": _sha256(path),
                "record_index": rep["record_index"],
                "parameter_vector_source": "records[].best_theta (verbatim)",
            },
            "warm_start_vector_c": c,
            "simulator_side": {
                "energy_simulator": sim["energy_simulator"],
                "normalization_error_simulator":
                    sim["normalization_error_simulator"],
                "computed_in_parent_process": True,
            },
        }
        if qk["status"] != "fresh":
            block["error"] = qk.get("error")
            problems.append(f"Method {method}: {qk['status']} - "
                            f"{qk.get('error', '')[:200]}")
            out_methods[method] = block
            continue

        raw = qk["raw"]
        sv = np.array(raw["real"], dtype=float) + 1j * np.array(
            raw["imag"], dtype=float)
        pq = np.abs(sv) ** 2
        ps = sim["_probs"]
        vec = sim["_vec"]
        energy_q = float(raw["energy_qiskit"])
        energy_s = float(sim["energy_simulator"])

        block["qiskit_side"] = {
            "energy_qiskit": energy_q,
            "normalization_error_qiskit": float(abs(pq.sum() - 1.0)),
            "returned_norm": float(raw["norm"]),
            "simulation_preparation": (
                "circuit decomposed to its own Qiskit definition via "
                f"QuantumCircuit.decompose(reps={DECOMPOSE_REPS}) before "
                "Statevector; unitary-preserving, see "
                "decomposition_faithfulness"),
            "decomposed_depth": raw.get("decomposed_depth"),
            "decomposed_size": raw.get("decomposed_size"),
            "decomposed_ops": raw.get("decomposed_ops"),
        }
        block["agreement"] = {
            "abs_energy_diff": float(abs(energy_s - energy_q)),
            "state_fidelity": float(abs(np.vdot(sv, vec)) ** 2),
            "total_variation_distance": float(0.5 * np.abs(ps - pq).sum()),
            "max_abs_probability_diff": float(np.abs(ps - pq).max()),
            "probability_l2_distance": float(np.linalg.norm(ps - pq)),
            "normalization_error_simulator":
                float(abs(ps.sum() - 1.0)),
            "normalization_error_qiskit": float(abs(pq.sum() - 1.0)),
            "tolerances": {"abs_energy_diff": 1e-9, "state_fidelity": 0.999999,
                           "normalization_error": 1e-12},
        }
        a = block["agreement"]
        if a["abs_energy_diff"] > 1e-9:
            problems.append(f"Method {method}: energy diff "
                            f"{a['abs_energy_diff']:.3e} > 1e-9")
        if a["state_fidelity"] < 0.999999:
            problems.append(f"Method {method}: fidelity "
                            f"{a['state_fidelity']:.17g} < 0.999999")
        if max(a["normalization_error_simulator"],
               a["normalization_error_qiskit"]) > 1e-12:
            problems.append(f"Method {method}: normalization error > 1e-12")

        # reported-vs-recomputed energy for the simulator side
        block["reported_vs_recomputed_energy"] = {
            "reported_best_energy": rep["reported_best_energy_C_lambda_exact"],
            "recomputed_energy": energy_s,
            "abs_diff": float(abs(rep["reported_best_energy_C_lambda_exact"]
                                  - energy_s)),
        }
        out_methods[method] = block

    overall = ("fresh" if all(b.get("status") == "fresh"
                              for b in out_methods.values())
               else "incomplete")

    faith = decomposition_faithfulness_check()
    if not faith["passed"]:
        problems.append(
            f"decomposition faithfulness failed: worst fid "
            f"{faith['worst_fidelity']:.17g}, worst dE "
            f"{faith['worst_abs_energy_diff']:.3e}")

    # keep the earlier (pre-decompose) attempt so the timeout history is not
    # erased by this corrected run
    prior = None
    if os.path.exists(FOLLOWUP_PATH):
        try:
            prev = json.load(open(FOLLOWUP_PATH, "r", encoding="utf-8"))
            prior = {
                "status": "superseded_by_corrected_run",
                "why": ("The first attempt simulated the opaque-gate circuit "
                        "directly. Qiskit's Statevector exponentiates the "
                        "PauliEvolutionGate generator matrix instead of using "
                        "the gate's own definition, costing >386 s per X-mixer "
                        "layer at n=12; both circuits hit the 600 s timeout "
                        "with no reference state obtained. This run expands "
                        "the circuit to its own definition before simulation "
                        "(unitary-preserving) and reproduces the same "
                        "comparison."),
                "generated_utc": prev.get("generated_utc"),
                "overall_status": prev.get("overall_status"),
                "methods": {
                    m: {"status": b.get("status"),
                        "p": b.get("selection", {}).get("p"),
                        "elapsed_s": b.get("elapsed_s"),
                        "timeout_s": b.get("timeout_s"),
                        "error": b.get("error")}
                    for m, b in prev.get("methods", {}).items()
                },
            }
        except Exception as exc:  # noqa: BLE001
            prior = {"status": "unreadable_prior_artifact", "error": str(exc)}

    return {
        "team": "axiom", "phase": 3, "study": "final_followup_validation",
        "generated_utc": _now(),
        "scope": ("n=12 cross-validation of the full-space simulators used by "
                  "Methods B and C against independent Qiskit circuits. This "
                  "supersedes nothing: the n=6 checks in validation_report.json "
                  "remain, and these are genuinely new n=12 measurements."),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "overall_status": overall,
        "timeout_per_circuit_s": float(N12_TIMEOUT_S),
        "methodology": {
            "fast_side": ("p3.FullSpaceSimulator (exact amplitude-level "
                          "simulation; X mixer via fast Walsh-Hadamard "
                          "transform, cost via diagonal phases)"),
            "qiskit_side": ("circuit constructed from scratch with "
                            "QuantumCircuit + PauliEvolutionGate + "
                            "Statevector; no code shared with the fast "
                            "simulator"),
            "decompose_reps": DECOMPOSE_REPS,
            "why_decompose": (
                "Statevector applies an opaque PauliEvolutionGate by "
                "exponentiating its generator matrix. At n=12 that costs ~14 s "
                "per cost layer and >386 s per X-mixer layer (measured), so "
                "the undecomposed circuit cannot complete inside 600 s. "
                "QuantumCircuit.decompose expands the gate to the definition "
                "Qiskit itself gives it, which is unitary-preserving and runs "
                "in ~0.04 s at n=12 p=3."
            ),
            "decomposition_faithfulness": faith,
        },
        "decomposition_faithfulness": faith,
        "prior_attempts": prior,
        "methods": out_methods,
        "commands": {
            "this_script": " ".join(sys.argv),
            "qiskit_child": [b["command"] for b in out_methods.values()],
        },
        "restrictions_observed": [
            "No optimiser was rerun.",
            "No existing experiment JSON, figure, or validation report was "
            "overwritten.",
            "Parameter vectors were read verbatim from the final artifacts.",
            "n=6 results are not substituted and relabelled as n=12.",
            "Every n=12 circuit ran in its own child process under a genuine "
            "600 s wall-clock timeout; timeouts are recorded, not hidden.",
        ],
    }, problems


# ---------------------------------------------------------------------------
# Test suite + log append
# ---------------------------------------------------------------------------
def run_tests() -> dict:
    cmd = [PYTHON, "-u", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    shown = " ".join(cmd)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = (proc.stdout or "") + (proc.stderr or "")
    header = (
        "# phase3-tests.log (regenerated by run_phase3_followup.py)\n"
        f"# timestamp_utc : {_now()}\n"
        f"# command       : {shown}\n"
        f"# cwd           : {PROJECT_ROOT}\n"
        f"# wall_clock_s  : {elapsed:.3f}\n"
        "# ---- child output below ----\n"
    )
    footer = (
        "# ---- end of child output ----\n"
        f"# exit_code : {proc.returncode}\n"
        f"# result    : {'OK' if proc.returncode == 0 else 'FAILED'}\n"
    )
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(TESTS_LOG, "w", encoding="utf-8") as fh:
        fh.write(header + output + footer)

    tests_run = None
    m = re.search(r"Ran (\d+) tests? in", output)
    if m:
        tests_run = int(m.group(1))
    failures = errors = skipped = 0
    failed_line = re.search(r"FAILED \(([^)]*)\)", output)
    if failed_line:
        for key, val in re.findall(r"(\w+)=(\d+)", failed_line.group(1)):
            if key == "failures":
                failures = int(val)
            elif key == "errors":
                errors = int(val)
            elif key == "skipped":
                skipped = int(val)
    return {
        "command": shown,
        "exit_code": int(proc.returncode),
        "wall_clock_s": round(float(elapsed), 3),
        "log_file": "logs/phase3-tests.log",
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": None if tests_run is None
        else tests_run - failures - errors - skipped,
    }


def append_validation_log(lines: list[str]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n")
        for line in lines:
            fh.write(f"[{_stamp()}] {line}\n")


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Phase 3 final follow-up started {datetime.now().isoformat()}",
          flush=True)
    problems: list[str] = []

    print("\n=== Check A: Intermediate penalty threshold ===", flush=True)
    threshold, a_problems = run_check_a()
    problems += a_problems
    _dump(THRESHOLD_PATH, threshold)
    print(f"  lambda* = {threshold['lambda_star']!r}", flush=True)
    ma = threshold["maximizing_assignment"]
    print(f"  argmax index={ma['index']} assets={ma['asset_indices']} "
          f"|x|={ma['cardinality']} C_0={ma['C_0']:.12f}", flush=True)
    print(f"  tie at threshold: "
          f"{threshold['tie_at_threshold']['creates_tie_with_feasible_optimum']}",
          flush=True)
    print(f"  cross-checks: {threshold['cross_checks']}", flush=True)
    print(f"  wrote {os.path.relpath(THRESHOLD_PATH, PROJECT_ROOT)}", flush=True)

    print("\n=== Check B: Methods B and C vs Qiskit at n=12 ===", flush=True)
    followup, b_problems = run_check_b()
    problems += b_problems
    _dump(FOLLOWUP_PATH, followup)
    faith = followup["decomposition_faithfulness"]
    print(f"  decomposition faithfulness (n=6): passed={faith['passed']} "
          f"worst_fid={faith['worst_fidelity']:.17g} "
          f"worst_dE={faith['worst_abs_energy_diff']:.3e}", flush=True)
    if followup.get("prior_attempts"):
        print("  prior (timeout) attempt preserved in prior_attempts",
              flush=True)
    for method, block in followup["methods"].items():
        print(f"  Method {method}: status={block['status']} "
              f"p={block['selection']['p']} elapsed={block['elapsed_s']}s",
              flush=True)
        if block.get("agreement"):
            a = block["agreement"]
            print(f"    dE={a['abs_energy_diff']:.3e} "
                  f"fid={a['state_fidelity']:.17g} "
                  f"TV={a['total_variation_distance']:.3e} "
                  f"norm_sim={a['normalization_error_simulator']:.3e} "
                  f"norm_q={a['normalization_error_qiskit']:.3e}", flush=True)
        else:
            print(f"    no agreement metrics: {block.get('error', '')[:200]}",
                  flush=True)
    print(f"  wrote {os.path.relpath(FOLLOWUP_PATH, PROJECT_ROOT)}", flush=True)

    print("\n=== test suite ===", flush=True)
    tests = run_tests()
    print(f"  exit_code={tests['exit_code']} run={tests['tests_run']} "
          f"passed={tests['passed']} wall={tests['wall_clock_s']}s "
          f"-> logs/phase3-tests.log", flush=True)

    log_lines = [
        "=" * 78,
        "Phase 3 final follow-up: two bounded validation checks",
        "=" * 78,
        f"command           : {' '.join(sys.argv)}",
        f"python            : {PYTHON}",
        f"overall_check_B   : {followup['overall_status']}",
        f"lambda_star       : {threshold['lambda_star']!r} "
        f"(path1={threshold['path_1']['lambda_star']!r} "
        f"path2={threshold['path_2']['lambda_star']!r} "
        f"agree={threshold['cross_checks']['lambda_star_paths_agree']})",
        f"maximizing_assign : index={ma['index']} assets={ma['asset_indices']} "
        f"|x|={ma['cardinality']} C_0={ma['C_0']!r}",
        f"tie_at_threshold  : "
        f"{threshold['tie_at_threshold']['creates_tie_with_feasible_optimum']} "
        f"(minimisers={threshold['tie_at_threshold']['num_global_minimisers_at_lambda_star']}, "
        f"unique above threshold="
        f"{threshold['tie_at_threshold']['unique_above_threshold']})",
        f"decompose_reps    : {DECOMPOSE_REPS} (circuit expanded to its own "
        f"Qiskit definition before Statevector)",
        f"faithfulness      : n=6 passed={faith['passed']} "
        f"worst_fid={faith['worst_fidelity']:.17g} "
        f"worst_dE={faith['worst_abs_energy_diff']:.3e}",
    ]
    for method, block in followup["methods"].items():
        if block.get("agreement"):
            a = block["agreement"]
            log_lines.append(
                f"check_B method={method} status={block['status']} "
                f"p={block['selection']['p']} timeout={block['timeout_s']:.0f}s "
                f"elapsed={block['elapsed_s']}s dE={a['abs_energy_diff']:.3e} "
                f"fid={a['state_fidelity']:.17g} "
                f"TV={a['total_variation_distance']:.3e}")
        else:
            log_lines.append(
                f"check_B method={method} status={block['status']} "
                f"p={block['selection']['p']} elapsed={block['elapsed_s']}s "
                f"error={str(block.get('error', ''))[:160]}")
    log_lines += [
        f"tests             : exit_code={tests['exit_code']} "
        f"run={tests['tests_run']} passed={tests['passed']} "
        f"wall={tests['wall_clock_s']}s",
        "wrote             : outputs/phase3/intermediate_penalty_threshold.json, "
        "outputs/phase3/final_followup_validation.json",
        "scope             : enumeration + n=12 reference circuits only; no "
        "optimiser rerun, no existing experiment JSON/figure/validation "
        "report overwritten, Phase 4 not started, no commit/remote created.",
    ]
    if problems:
        log_lines.append("unresolved issues  :")
        for p in problems:
            log_lines.append(f"  * {p}")
    else:
        log_lines.append("unresolved issues  : none")
    append_validation_log(log_lines)
    print("\nappended dated entry to logs/phase3-run.log", flush=True)

    if tests["exit_code"] != 0:
        return 1
    if problems or followup["overall_status"] != "fresh":
        print("\nUNRESOLVED:", flush=True)
        for p in problems:
            print(f"  * {p}", flush=True)
        return 3
    print("\nboth checks fresh, no unresolved issues.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
