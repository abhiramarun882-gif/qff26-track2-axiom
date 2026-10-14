"""Phase 1 runner: compute all Phase 1 diagnostics and write JSON artifacts.

Run from the project root with the project interpreter:

    .venv\\Scripts\\python.exe run_phase1.py

Writes:
    outputs/phase1/beginner_results.json
    outputs/phase1/intermediate_results.json
    outputs/phase1/validation_report.json
    logs/phase1-tests.log
"""

from __future__ import annotations

import importlib
import io
import json
import os
import platform
import sys
import unittest
from datetime import datetime, timezone

import numpy as np

import phase1_core as core

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase1")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

BIT_ORDER_CONVENTION = (
    "asset i <-> qubit i; state index s = sum_i x_i 2^i (qubit 0 = least "
    "significant bit). Qiskit Pauli strings: rightmost character is qubit 0. "
    "Operator matrix index s must equal cost_penalized(index_to_assets(s, n))."
)


def _pkg_versions():
    names = ["qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_algorithms",
             "numpy", "scipy", "matplotlib", "rustworkx"]
    out = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unavailable ({exc})"
    return out


def _qubo_max_error(n, K, mu, Sigma, q, lam):
    c, a, B = core.qubo_coefficients(mu, Sigma, q, K, lam)
    if n == 6:
        xs = [core.index_to_assets(s, n) for s in range(2 ** n)]
    else:
        xs = [np.asarray(x) for x in core.feasible_assignments(n, K)]
        rng = np.random.default_rng(101)
        need = 500
        while need > 0:
            s = int(rng.integers(0, 2 ** n))
            x = core.index_to_assets(s, n)
            if int(x.sum()) != K:
                xs.append(x)
                need -= 1
    max_err = 0.0
    for x in xs:
        direct = core.cost_penalized(x, mu, Sigma, K, lam, q)
        qubo = core.cost_from_qubo(x, c, a, B)
        max_err = max(max_err, abs(direct - qubo))
    return float(max_err), len(xs)


def _pauli_max_error(n, K, mu, Sigma, q, lam):
    op, c_I, h, J = core.qubo_to_pauli(mu, Sigma, q, K, lam)
    diag = core.operator_diagonal(op)
    if n == 6:
        indices = list(range(2 ** n))
    else:
        indices = [core.assets_to_index(x) for x in core.feasible_assignments(n, K)]
        rng = np.random.default_rng(202)
        need = 500
        while need > 0:
            s = int(rng.integers(0, 2 ** n))
            x = core.index_to_assets(s, n)
            if int(x.sum()) != K:
                indices.append(s)
                need -= 1
    max_err = 0.0
    for s in indices:
        x = core.index_to_assets(s, n)
        direct = core.cost_penalized(x, mu, Sigma, K, lam, q)
        max_err = max(max_err, abs(diag[s] - direct))
    return float(max_err), len(indices)


def _run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.join(PROJECT_ROOT, "tests"),
                            top_level_dir=PROJECT_ROOT)
    buf = io.StringIO()
    runner = unittest.TextTestRunner(stream=buf, verbosity=2)
    result = runner.run(suite)
    text = buf.getvalue()
    print(text)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "phase1-tests.log"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors)
        - len(result.skipped),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }, (not result.wasSuccessful())


def _mixer_diagnostics():
    from qiskit.quantum_info import Operator
    import math

    hermiticity = {}
    for n in (4, 6):
        op = core.build_xy_ring_mixer(n)
        diff = (op - op.adjoint()).to_matrix(sparse=True)
        hermiticity[f"n={n}"] = float(abs(diff).max() if diff.nnz else 0.0)
    hm6 = core.build_xy_ring_mixer(6)
    zsum6 = core.excitation_number_operator(6)
    comm = Operator(hm6).data @ Operator(zsum6).data - Operator(zsum6).data @ Operator(hm6).data
    return {
        "hermiticity_max_abs": hermiticity,
        "commutator_norm_n6": float(np.linalg.norm(comm)),
        "excitation_conserved": bool(np.linalg.norm(comm) <= core.COMMUTATOR_ATOL),
        "not_x_only": bool(any("Y" in l for l in hm6.paulis.to_labels())),
        "ring_edges_n6": 6,
        "n2_doubling_documented": True,
        "subspace_dims": {"weight3_n6": int(math.comb(6, 3)),
                          "weight6_n12": int(math.comb(12, 6))},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lam_ref = 1.0

    beginner = core.generate_instance(6, 3)
    intermediate = core.generate_instance(12, 6)

    b_props = core.check_instance_properties(beginner)
    i_props = core.check_instance_properties(intermediate)

    b_qubo_err, b_qubo_n = _qubo_max_error(6, 3, beginner.mu, beginner.Sigma, beginner.q, lam_ref)
    i_qubo_err, i_qubo_n = _qubo_max_error(12, 6, intermediate.mu, intermediate.Sigma, intermediate.q, lam_ref)
    b_pauli_err, b_pauli_n = _pauli_max_error(6, 3, beginner.mu, beginner.Sigma, beginner.q, lam_ref)
    i_pauli_err, i_pauli_n = _pauli_max_error(12, 6, intermediate.mu, intermediate.Sigma, intermediate.q, lam_ref)

    b_bench = core.exact_benchmark(6, 3, beginner.mu, beginner.Sigma, beginner.q, lam_ref)
    i_bench = core.exact_benchmark(12, 6, intermediate.mu, intermediate.Sigma, intermediate.q, lam_ref)

    b_state = core.weight_state_support_counts(6, 3)
    i_state = core.weight_state_support_counts(12, 6)

    mixer = _mixer_diagnostics()

    test_stats, tests_failed = _run_tests()

    beginner_results = {
        "team": "axiom",
        "phase": 1,
        "level": "beginner",
        "instance": beginner.to_dict(),
        "lambda_ref": lam_ref,
        "qubo_max_abs_error": b_qubo_err,
        "qubo_assignments_checked": b_qubo_n,
        "pauli_max_abs_error": b_pauli_err,
        "pauli_assignments_checked": b_pauli_n,
        "instance_properties": b_props,
        "benchmark": b_bench,
        "initial_state": b_state,
        "bit_order_convention": BIT_ORDER_CONVENTION,
    }
    intermediate_results = {
        "team": "axiom",
        "phase": 1,
        "level": "intermediate",
        "instance": intermediate.to_dict(),
        "lambda_ref": lam_ref,
        "qubo_max_abs_error": i_qubo_err,
        "qubo_assignments_checked": i_qubo_n,
        "pauli_max_abs_error": i_pauli_err,
        "pauli_assignments_checked": i_pauli_n,
        "instance_properties": i_props,
        "benchmark": i_bench,
        "initial_state": i_state,
        "bit_order_convention": BIT_ORDER_CONVENTION,
    }

    validation_report = {
        "team": "axiom",
        "phase": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _pkg_versions(),
        "seed": core.DEFAULT_SEED,
        "q": core.DEFAULT_Q,
        "data_generation": {
            "procedure": "rng=np.random.default_rng(seed); mu=uniform(.05,.25,n); "
                         "A=normal(0,.1,(n,n)); Sigma=A@A.T (no symmetrisation)",
            "beginner": {"n": 6, "K": 3, "feasible": b_bench["feasible_count"]},
            "intermediate": {"n": 12, "K": 6, "feasible": i_bench["feasible_count"]},
        },
        "tests": test_stats,
        "numeric_agreement": {
            "qubo_vs_direct": {"beginner_max_abs_error": b_qubo_err,
                               "intermediate_max_abs_error": i_qubo_err,
                               "tolerance": core.OBJECTIVE_ATOL},
            "pauli_vs_direct": {"beginner_max_abs_error": b_pauli_err,
                                "intermediate_max_abs_error": i_pauli_err,
                                "tolerance": core.MATRIX_ATOL},
        },
        "enumeration": {
            "beginner_feasible": b_bench["feasible_count"],
            "beginner_expected": b_bench["expected_feasible_count"],
            "intermediate_feasible": i_bench["feasible_count"],
            "intermediate_expected": i_bench["expected_feasible_count"],
            "independent_agreement_beginner": b_bench["independent_agreement"],
            "independent_agreement_intermediate": i_bench["independent_agreement"],
        },
        "exact_optima": {
            "beginner": {
                "cost": b_bench["optimal_unpenalized_cost"],
                "asset_indices": b_bench["optimal_asset_indices"],
                "assignment": b_bench["optimal_assignment"],
                "num_ties": b_bench["num_optimal_ties"],
            },
            "intermediate": {
                "cost": i_bench["optimal_unpenalized_cost"],
                "asset_indices": i_bench["optimal_asset_indices"],
                "assignment": i_bench["optimal_assignment"],
                "num_ties": i_bench["num_optimal_ties"],
            },
        },
        "xy_mixer": mixer,
        "initial_state": {
            "beginner": b_state,
            "intermediate": i_state,
        },
        "bit_order_convention": BIT_ORDER_CONVENTION,
        "known_limitations": [
            "DickeStatePreparation removed in Qiskit 2.x; uniform fixed-weight "
            "state is built as an explicit amplitude vector via StatePreparation.",
            "The XY ring mixer uses the literal periodic expression; for n=2 the "
            "single edge is listed twice and the effective coupling doubles "
            "(documented, not deduplicated).",
            "Cost is a synthetic benchmark, not a market-data model.",
        ],
    }

    def _dump(path, obj):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False)

    _dump(os.path.join(OUT_DIR, "beginner_results.json"), beginner_results)
    _dump(os.path.join(OUT_DIR, "intermediate_results.json"), intermediate_results)
    _dump(os.path.join(OUT_DIR, "validation_report.json"), validation_report)

    print("=== Phase 1 summary ===")
    print(f"beginner  n=6 K=3 feasible={b_bench['feasible_count']} "
          f"opt={b_bench['optimal_unpenalized_cost']:.6f} "
          f"assets={b_bench['optimal_asset_indices']}")
    print(f"intermediate n=12 K=6 feasible={i_bench['feasible_count']} "
          f"opt={i_bench['optimal_unpenalized_cost']:.6f} "
          f"assets={i_bench['optimal_asset_indices']}")
    print(f"qubo max err  B={b_qubo_err:.3e}  I={i_qubo_err:.3e}")
    print(f"pauli max err B={b_pauli_err:.3e}  I={i_pauli_err:.3e}")
    print(f"tests: {test_stats}")
    return 1 if tests_failed else 0


if __name__ == "__main__":
    sys.exit(main())
