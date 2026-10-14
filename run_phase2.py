"""Phase 2 runner: depth study, penalty study, figures, JSON artifacts, tests.

Run from the project root with the project interpreter:

    .venv\\Scripts\\python.exe run_phase2.py

Writes:
    outputs/phase2/depth_results.json
    outputs/phase2/penalty_results.json
    outputs/phase2/experiment_summary.json
    outputs/phase2/validation_report.json
    outputs/phase2/figures/*.png
    logs/phase2-tests.log
    logs/phase2-run.log
"""

from __future__ import annotations

import copy
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
import phase2_qaoa as qaoa
import phase2_experiments as exp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase2")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

N_STARTS = 8
# COBYLA is started at the prompt's initial cap of 250 evaluations per start.
# At 250, all p=3 and p=4 starts terminate on the evaluation cap (status 3)
# without converging, so the cap is raised to 1500 to let most starts converge;
# termination status and the success fraction are still recorded per start.
MAXITER = 1500
BASE_SEED = core.DEFAULT_SEED
RANDOM_BASELINE_DRAWS = 10000

EXPECTED_OPTIMUM = -0.5030146902647753
EXPECTED_ASSETS = [1, 2, 5]


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
    with open(os.path.join(LOG_DIR, "phase2-tests.log"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors)
        - len(result.skipped),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }, (not result.wasSuccessful())


def _circuit_validation(inst):
    """Independent circuit/energy/sampling validation numbers for the report."""
    lam = qaoa.PRIMARY_LAMBDA
    cost_op = qaoa.cost_operator(inst, lam)
    rng = np.random.default_rng(4242)
    max_ansatz_err = 0.0
    max_diag_err = 0.0
    max_prob_err = 0.0
    max_fast_energy_err = 0.0
    min_fast_fidelity = 1.0
    diag = core.operator_diagonal(cost_op)
    for p in qaoa.DEPTHS:
        qc = qaoa.build_qaoa_circuit(inst.n, p, cost_op, qaoa.build_x_mixer(inst.n))
        sim = qaoa.FastQAOASimulator(inst.n, p, diag)
        for _ in range(3):
            theta = exp._jsonable(rng.uniform(0.0, 2 * np.pi, size=2 * p))
            e = qaoa.qaoa_energy(qc, cost_op, theta)
            max_ansatz_err = max(max_ansatz_err,
                                 abs(e - qaoa.qaoa_ansatz_energy(inst.n, p, cost_op, theta)))
            vec = qaoa.qaoa_statevector(qc, theta).data
            max_diag_err = max(max_diag_err,
                               abs(e - qaoa.qaoa_energy_from_vector(cost_op, vec)))
            max_prob_err = max(max_prob_err,
                               abs(float(qaoa.exact_probabilities(qc, theta).sum()) - 1.0))
            fast_vec = sim.statevector(theta)
            min_fast_fidelity = min(min_fast_fidelity,
                                    abs(np.vdot(vec, fast_vec)) ** 2)
            max_fast_energy_err = max(max_fast_energy_err,
                                      abs(e - sim.energy(theta)))
    return {
        "num_qubits": inst.n,
        "parameter_count_by_depth": {f"p={p}": 2 * p for p in qaoa.DEPTHS},
        "max_abs_error_explicit_vs_qaoa_ansatz": float(max_ansatz_err),
        "max_abs_error_statevector_vs_diagonal_contraction": float(max_diag_err),
        "max_abs_probability_normalisation_error": float(max_prob_err),
        "max_abs_error_fast_simulator_vs_qiskit_energy": float(max_fast_energy_err),
        "min_fidelity_fast_simulator_vs_qiskit": float(min_fast_fidelity),
        "energy_tolerance": qaoa.ENERGY_ATOL,
        "sampling_method": exp.SAMPLING_METHOD,
        "bit_order_convention": (
            "asset i <-> qubit i; state index s = sum_i x_i 2^i (qubit 0 = LSB); "
            "Qiskit Pauli rightmost char is qubit 0"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    inst = core.generate_instance(6, 3)
    bench = core.exact_benchmark(6, 3, inst.mu, inst.Sigma, inst.q)

    # --- hard guard: reference optimum must match Phase 1 exactly ---
    if not np.isclose(bench["optimal_unpenalized_cost"], EXPECTED_OPTIMUM, atol=1e-9):
        raise SystemExit(f"BLOCKER: exact optimum {bench['optimal_unpenalized_cost']} "
                         f"!= expected {EXPECTED_OPTIMUM}")
    if sorted(bench["optimal_asset_indices"]) != EXPECTED_ASSETS:
        raise SystemExit(f"BLOCKER: optimal assets {bench['optimal_asset_indices']} "
                         f"!= expected {EXPECTED_ASSETS}")

    print("=== Phase 2: standard penalized QAOA baseline (Beginner n=6, K=3) ===")
    print(f"exact optimum C_best={bench['best_feasible_objective']:.12f} "
          f"assets={bench['optimal_asset_indices']} "
          f"C_worst={bench['worst_feasible_objective']:.12f}")

    depth = exp.run_depth_study(inst, qaoa.DEPTHS, qaoa.PRIMARY_LAMBDA,
                                N_STARTS, BASE_SEED, MAXITER, bench)
    penalty = exp.run_penalty_study(inst, qaoa.PENALTIES, qaoa.PRIMARY_DEPTH,
                                    N_STARTS, BASE_SEED, MAXITER, bench)
    exact_pen = qaoa.exact_penalty_analysis(inst, qaoa.PENALTIES)
    baseline = exp.random_feasible_baseline(inst, bench, RANDOM_BASELINE_DRAWS,
                                            seed=BASE_SEED + 31)

    figures = exp.make_figures(depth, penalty, exact_pen, FIG_DIR)

    # --- select best results by a documented criterion ---
    best_depth = min(depth["records"], key=lambda r: r["best_energy"])
    threshold = exact_pen["penalty_threshold"]
    feasible_pen = [r for r in penalty["records"] if r["lambda"] >= threshold - 1e-12]
    rec_pen = min(feasible_pen, key=lambda r: r["lambda"]) if feasible_pen \
        else penalty["records"][-1]

    versions = _pkg_versions()
    circuit_val = _circuit_validation(inst)

    instance_block = {
        "level": "beginner",
        "n": inst.n, "K": inst.K, "q": inst.q, "seed": inst.seed,
        "mu": [float(v) for v in inst.mu],
        "Sigma": [[float(v) for v in row] for row in inst.Sigma],
    }
    exact_block = {
        "C_best": bench["best_feasible_objective"],
        "C_worst": bench["worst_feasible_objective"],
        "optimal_asset_indices": bench["optimal_asset_indices"],
        "optimal_assignment": bench["optimal_assignment"],
        "feasible_count": bench["feasible_count"],
        "num_optimal_ties": bench["num_optimal_ties"],
        "expected_optimum": EXPECTED_OPTIMUM,
        "optimum_match": bool(np.isclose(bench["optimal_unpenalized_cost"],
                                         EXPECTED_OPTIMUM, atol=1e-9)),
    }

    depth_out = {
        "team": "axiom", "phase": 2, "study": "depth",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": {"depths": list(qaoa.DEPTHS), "lambda": qaoa.PRIMARY_LAMBDA,
                     "n_starts": N_STARTS, "optimizer": qaoa.OPTIMIZER_NAME,
                     "maxiter": MAXITER, "rhobeg": qaoa.DEFAULT_RHOBEG,
                     "initialisation_range": exp.INIT_RANGE,
                     "base_seed": BASE_SEED, "shots": qaoa.SHOTS,
                     "sampling_method": exp.SAMPLING_METHOD},
        "records": depth["records"],
        "package_versions": versions,
    }
    penalty_out = {
        "team": "axiom", "phase": 2, "study": "penalty",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": {"penalties": list(qaoa.PENALTIES), "p": qaoa.PRIMARY_DEPTH,
                     "n_starts": N_STARTS, "optimizer": qaoa.OPTIMIZER_NAME,
                     "maxiter": MAXITER, "rhobeg": qaoa.DEFAULT_RHOBEG,
                     "initialisation_range": exp.INIT_RANGE,
                     "base_seed": BASE_SEED, "shots": qaoa.SHOTS,
                     "sampling_method": exp.SAMPLING_METHOD},
        "records": penalty["records"],
        "exact_penalty_analysis": exact_pen,
        "package_versions": versions,
    }

    def _link(rec):
        return {
            "p": rec.get("p"), "lambda": rec["lambda"],
            "best_energy": rec.get("best_energy", rec.get("best_penalized_energy")),
            "best_theta": rec["best_theta"],
            "best_start_index": rec["best_start_index"],
            "starts_summary": rec["starts_summary"],
            "sampling": rec["sampling"],
        }

    summary = {
        "team": "axiom", "phase": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": {"depths": list(qaoa.DEPTHS), "penalties": list(qaoa.PENALTIES),
                     "primary_lambda": qaoa.PRIMARY_LAMBDA,
                     "primary_depth": qaoa.PRIMARY_DEPTH,
                     "n_starts": N_STARTS, "optimizer": qaoa.OPTIMIZER_NAME,
                     "maxiter": MAXITER, "shots": qaoa.SHOTS},
        "best_by_criterion": {
            "criterion": "lowest best penalized expectation <H_C> over all starts",
            "best_depth": _link(best_depth),
            "recommended_penalty": {
                "criterion": ("smallest tested lambda that is >= the exact penalty "
                              "threshold (so the feasible optimum is a global "
                              "penalized minimum)"),
                "penalty_threshold": threshold,
                "selected": _link(rec_pen),
            },
        },
        "random_feasible_baseline": baseline,
        "figures": [os.path.relpath(p, PROJECT_ROOT) for p in figures],
        "package_versions": versions,
        "reproduction": {
            "command": ".venv\\Scripts\\python.exe run_phase2.py",
            "python": sys.version.split()[0],
        },
    }

    validation = {
        "team": "axiom", "phase": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "package_versions": versions,
        "tests": None,  # filled after test run
        "circuit_validation": circuit_val,
        "exact_penalty_analysis": exact_pen,
        "exact_reference": exact_block,
        "sampling_note": ("Sampled results are draws from the exact statevector "
                          "distribution via a seeded multinomial; they are NOT "
                          "hardware measurements. Exact-statevector metrics are "
                          "reported alongside and kept separate."),
        "metric_definitions": {
            "C_best": "exact optimal feasible unpenalized cost",
            "C_worst": "exact worst feasible unpenalized cost",
            "feasible_normalized_score":
                "(C_worst - E[C0 | feasible]) / (C_worst - C_best); null if no "
                "feasible samples or C_worst == C_best",
            "optimal_solution_probability":
                "probability mass on any globally optimal feasible portfolio",
        },
        "known_limitations": [
            "Single synthetic 6-asset instance; no claim of quantum advantage.",
            "Exhaustive search is the true reference for this small instance.",
            "Sampled metrics come from exact-statevector multinomial draws, not hardware.",
            "QAOA parameters are classically optimised with COBYLA (no gradient information).",
        ],
    }

    # depth_results / penalty_results already contain full records.
    depth_out_final = copy.deepcopy(depth_out)
    penalty_out_final = copy.deepcopy(penalty_out)

    # --- run tests before finalising report ---
    test_stats, tests_failed = _run_tests()
    validation["tests"] = test_stats

    def _dump(path, obj):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(exp._jsonable(obj), fh, indent=2)

    _dump(os.path.join(OUT_DIR, "depth_results.json"), depth_out_final)
    _dump(os.path.join(OUT_DIR, "penalty_results.json"), penalty_out_final)
    _dump(os.path.join(OUT_DIR, "experiment_summary.json"), summary)
    _dump(os.path.join(OUT_DIR, "validation_report.json"), validation)

    print("--- depth study (lambda=1.0) ---")
    for r in depth["records"]:
        sm = r["sampling"]["sampled_metrics"]
        print(f"p={r['p']}: E*={r['best_energy']:.6f} "
              f"feas={sm['feasibility_rate']:.3f} "
              f"P(opt)={sm['optimal_solution_probability']:.4f} "
              f"score={sm['feasible_normalized_score']} "
              f"nfev(best)={r['best_start_nfev']} wall={r['total_wall_clock_s']:.2f}s "
              f"succ={r['starts_summary']['num_successful']}/{N_STARTS}")
    print("--- penalty study (p=2) ---")
    for r in penalty["records"]:
        em = r["sampling"]["exact_statevector_metrics"]
        print(f"lambda={r['lambda']}: E*={r['best_penalized_energy']:.6f} "
              f"exactFeas={em['feasibility_rate']:.3f} "
              f"exactP(opt)={em['optimal_solution_probability']:.4f} "
              f"wall={r['total_wall_clock_s']:.2f}s")
    print(f"exact penalty threshold = {threshold:.6f}")
    print(f"best depth by energy: p={best_depth['p']} E*={best_depth['best_energy']:.6f}")
    print(f"recommended penalty: lambda={rec_pen['lambda']}")
    print(f"tests: {test_stats}")
    print(f"figures: {len(figures)} written to {os.path.relpath(FIG_DIR, PROJECT_ROOT)}")
    return 1 if tests_failed else 0


if __name__ == "__main__":
    sys.exit(main())
