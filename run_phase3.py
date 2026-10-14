"""Phase 3 runner: methods, baselines, comparison, figures, JSON, tests.

Team axiom - Qiskit Fall Fest 2026 - Track 2 (Intermediate n=12, K=6).

Run from the project root with the project interpreter:

    .venv\\Scripts\\python.exe run_phase3.py

Writes (all under outputs/phase3/):
    continuous_relaxation.json
    method_a_xy_qaoa.json
    method_b_warm_start.json
    method_c_penalty_qaoa.json
    classical_baselines.json
    comparison_summary.json
    figure_data.json
    validation_report.json
    figures/*.png
and logs/phase3-run.log, logs/phase3-tests.log.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import platform
import sys
import time
import unittest
from datetime import datetime, timezone

import numpy as np

import phase1_core as core
import phase2_qaoa as q2
import phase3_methods as p3
import phase3_experiments as exp

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase3")
FIG_DIR = os.path.join(OUT_DIR, "figures")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

N, K = 12, 6
EXPECTED_OPTIMUM = -1.0663920923466492
EXPECTED_ASSETS = [1, 5, 6, 8, 10, 11]

DEPTHS = p3.DEPTHS
N_STARTS = p3.N_STARTS
MAXITER = p3.MAXITER
LAMBDA = p3.LAMBDA
SHOTS = p3.SHOTS
BASE_SEED = core.DEFAULT_SEED
SA_RESTARTS = p3.SA_RESTARTS
SA_STEPS = 2000
SA_T0, SA_T1 = 0.2, 0.002
RELAX_SEEDS = (11, 22, 33)


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


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(exp._jsonable(obj), fh, indent=2)
    print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}")


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
    with open(os.path.join(LOG_DIR, "phase3-tests.log"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return {
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors)
        - len(result.skipped),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
    }, (not result.wasSuccessful())


# ---------------------------------------------------------------------------
# Continuous relaxation (multistart, keep best)
# ---------------------------------------------------------------------------
def _run_relaxation(inst):
    points = [np.full(inst.n, inst.K / inst.n)]
    for s in RELAX_SEEDS:
        points.append(np.random.default_rng(s).uniform(0.2, 0.8, size=inst.n))
    runs = []
    for i, c0 in enumerate(points):
        r = p3.continuous_relaxation(inst, LAMBDA, c0=c0)
        r["init_kind"] = "uniform K/n" if i == 0 else f"random(seed={RELAX_SEEDS[i-1]})"
        r["rounding"] = p3.round_relaxation(inst, r["final_point"])
        runs.append(r)
    best = min(runs, key=lambda r: r["objective"])
    return {
        "lambda": float(LAMBDA),
        "num_restarts": len(runs),
        "initial_points": [np.asarray(r["initial_point"]).round(6).tolist()
                           for r in runs],
        "best_run": best,
        "runs": runs,
        "note": ("SLSQP minimises the penalised objective over c in [0,1]^n. The "
                 "relaxation yields a lower bound (objective <= true feasible "
                 "minimum) and a per-asset importance vector, useful only as a "
                 "warm-start / initial-rotation heuristic. It guarantees neither "
                 "integrality nor exact cardinality; rounding+repair is a "
                 "heuristic that recovers a weight-K portfolio but may be "
                 "suboptimal and can differ from the relaxation's own point."),
    }


# ---------------------------------------------------------------------------
# Validation vs independent Qiskit circuits
# ---------------------------------------------------------------------------
def _validate_method_a(inst, depths=(1, 3)):
    n, K = inst.n, inst.K
    cost_op0 = core.qubo_to_pauli(inst.mu, inst.Sigma, inst.q, K, 0.0)[0]
    diag0 = core.operator_diagonal(cost_op0)
    feas = p3.feasible_indices(n, K)
    rng = np.random.default_rng(20260303)
    out = []
    for p in depths:
        theta = rng.uniform(0.0, 2 * np.pi, size=2 * p)
        sim = p3.WeightSubspaceSimulator(n, K, diag0, p)
        sub = sim.statevector(theta)
        qc = p3.build_method_a_circuit(inst, cost_op0, p)
        from qiskit.quantum_info import Statevector
        full = Statevector(q2.bind_theta(qc, theta)).data
        sub_q = full[feas]
        norm = np.linalg.norm(sub_q)
        fid = abs(np.vdot(sub_q / norm, sub / np.linalg.norm(sub))) ** 2
        # energy of the full state wrt objective-only Hamiltonian, feasible slice
        e_q = float(np.sum(np.abs(sub_q) ** 2 * diag0[feas]) / norm ** 2)
        out.append({
            "p": int(p),
            "theta_seed": 20260303,
            "fidelity_subspace_vs_qiskit": float(fid),
            "energy_subspace": float(sim.energy(theta)),
            "energy_qiskit_feasible_conditioned": e_q,
            "abs_energy_diff": float(abs(sim.energy(theta) - e_q)),
            "infeasible_mass_qiskit": float(1.0 - norm ** 2),
            "qiskit_circuit_depth": int(qc.circuit.depth()),
            "state_preparation_note": ("StatePreparation is synthesised by Qiskit "
                                       "for this one-off reference; not used in loops"),
        })
    return out


def compare_full_space(inst, method, theta, c=None, lam=LAMBDA):
    """Compare p3.FullSpaceSimulator against an independent Qiskit circuit.

    This is the single reference-comparison routine for the full-space
    simulators used by Methods B and C. It is reused both by the n=6 phase-3
    validation (via :func:`_validate_full_space`) and by the n=12 final
    follow-up check, so the methodology stays identical at both sizes.

    Returns a dict of agreement metrics. The Qiskit circuit is built from
    scratch (initialisation + ``PauliEvolutionGate`` per layer) and never
    shares code with the fast simulator.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector

    p = len(theta) // 2
    cost_op = q2.cost_operator(inst, lam)
    diag = core.operator_diagonal(cost_op)
    if method == "B":
        if c is None:
            raise ValueError("method B requires the warm-start vector c")
        init_amp = p3.product_state_amplitudes(c)
    else:
        init_amp = p3.uniform_plus_amplitudes(inst.n)

    sim = p3.FullSpaceSimulator(inst.n, p, diag, init_amp)
    vec = sim.statevector(theta)

    qc = QuantumCircuit(inst.n)
    if method == "B":
        for i in range(inst.n):
            qc.ry(2 * np.arcsin(np.sqrt(c[i])), i)
    else:
        qc.h(range(inst.n))
    for k in range(p):
        qc.append(PauliEvolutionGate(cost_op, time=theta[k]), range(inst.n))
        qc.append(PauliEvolutionGate(q2.build_x_mixer(inst.n),
                                     time=theta[p + k]), range(inst.n))
    sv = Statevector(qc).data

    ps = np.abs(vec) ** 2
    pq = np.abs(sv) ** 2
    return {
        "p": int(p),
        "fidelity_vs_qiskit": float(abs(np.vdot(sv, vec)) ** 2),
        "abs_energy_diff": float(abs(sim.energy(theta)
                                      - q2.qaoa_energy_from_vector(cost_op, sv))),
        "energy_simulator": float(sim.energy(theta)),
        "energy_qiskit": float(q2.qaoa_energy_from_vector(cost_op, sv)),
        "total_variation_distance": float(0.5 * np.abs(ps - pq).sum()),
        "max_abs_probability_diff": float(np.abs(ps - pq).max()),
        "normalization_error_simulator": float(abs(ps.sum() - 1.0)),
        "normalization_error_qiskit": float(abs(pq.sum() - 1.0)),
    }


def _validate_full_space(inst6, method):
    """n=6 reference check; keeps the original seeded parameter sequence."""
    rng = np.random.default_rng(991)
    out = []
    if method == "B":
        c = rng.uniform(0.1, 0.9, size=inst6.n)
    else:
        c = None
    for p in (1, 2, 3):
        theta = rng.uniform(0.0, 2 * np.pi, size=2 * p)
        out.append(compare_full_space(inst6, method, theta, c=c))
    return out


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def _compact(rec):
    s = rec["sampling"]
    sm, em = s["sampled_metrics"], s["exact_statevector_metrics"]
    return {
        "p": rec["p"],
        "lambda": rec.get("lambda"),
        "best_energy_method_native": rec.get(
            "best_energy_C0_exact", rec.get("best_energy_C_lambda_exact")),
        "best_feasible_sampled": sm["best_feasible_unpenalized_cost"],
        "best_feasible_exact": em["best_feasible_unpenalized_cost"],
        "feasibility_rate_sampled": sm["feasibility_rate"],
        "feasibility_rate_exact": em["feasibility_rate"],
        "optimal_prob_sampled": sm["optimal_solution_probability"],
        "optimal_prob_exact": em["optimal_solution_probability"],
        "mean_c0_feasible_exact": em["mean_c0_conditioned_on_feasible"],
        "expected_c0_exact": em["expected_unpenalized_cost"],
        "score_exact": em["feasible_normalized_score"],
        "runtime_s": rec["optimization_wall_clock_s"],
        "num_successful": rec["starts_summary"]["num_successful"],
        "num_starts": rec["starts_summary"]["num_starts"],
        "total_nfev": sum(x["nfev"] for x in rec["starts"]),
    }


def _best_over(records, key):
    vals = [r[key] for r in records if r[key] is not None]
    return min(vals) if vals else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    t_start = time.perf_counter()
    log("=== Phase 3: quantum methods vs classical baselines "
        "(Intermediate n=12, K=6) ===")

    inst = core.generate_instance(N, K)
    bench = core.exact_benchmark(N, K, inst.mu, inst.Sigma, inst.q)
    if not np.isclose(bench["optimal_unpenalized_cost"], EXPECTED_OPTIMUM, atol=1e-9):
        raise SystemExit(f"BLOCKER: optimum {bench['optimal_unpenalized_cost']} != "
                         f"{EXPECTED_OPTIMUM}")
    if sorted(bench["optimal_asset_indices"]) != EXPECTED_ASSETS:
        raise SystemExit(f"BLOCKER: assets {bench['optimal_asset_indices']} != "
                         f"{EXPECTED_ASSETS}")
    C_best = float(bench["best_feasible_objective"])
    C_worst = float(bench["worst_feasible_objective"])
    log(f"exact optimum C_best={C_best:.12f} assets={bench['optimal_asset_indices']} "
        f"C_worst={C_worst:.12f} ties={bench['num_optimal_ties']}")

    versions = _pkg_versions()
    instance_block = {
        "level": "intermediate", "n": inst.n, "K": inst.K, "q": inst.q,
        "seed": inst.seed, "mu": [float(v) for v in inst.mu],
        "Sigma": [[float(v) for v in row] for row in inst.Sigma],
    }
    exact_block = {
        "C_best": C_best, "C_worst": C_worst,
        "optimal_asset_indices": bench["optimal_asset_indices"],
        "optimal_assignment": bench["optimal_assignment"],
        "feasible_count": bench["feasible_count"],
        "num_optimal_ties": bench["num_optimal_ties"],
        "expected_optimum": EXPECTED_OPTIMUM,
        "optimum_match": True,
    }
    settings = {
        "depths": list(DEPTHS), "lambda": LAMBDA, "n_starts": N_STARTS,
        "optimizer": q2.OPTIMIZER_NAME, "maxiter": MAXITER,
        "rhobeg": q2.DEFAULT_RHOBEG, "base_seed": BASE_SEED, "shots": SHOTS,
        "warm_start_seed": p3.WARM_START_SEED,
        "sa": {"restarts": SA_RESTARTS, "steps_per_restart": SA_STEPS,
               "T0": SA_T0, "T1": SA_T1, "schedule": "geometric",
               "base_seed": 777},
        "penalty_diagnostic": list(p3.PRIMARY_PENALTY_DIAGNOSTIC),
        "sampling_note": ("Sampled results are seeded multinomial draws from the "
                          "exact statevector distribution, NOT hardware "
                          "measurements."),
    }

    # --- 1. continuous relaxation ---
    log("\n--- continuous relaxation (SLSQP, penalised objective) ---")
    relax = _run_relaxation(inst)
    relax["gap_to_optimum"] = relax["best_run"]["objective"] - C_best
    relax_out = {
        "team": "axiom", "phase": 3, "study": "continuous_relaxation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": {"lambda": LAMBDA, "optimizer": "scipy SLSQP",
                     "bounds": "[0,1]^n", "n_restarts": len(relax["runs"])},
        "relaxation": relax, "package_versions": versions,
    }
    _dump(os.path.join(OUT_DIR, "continuous_relaxation.json"), relax_out)
    br = relax["best_run"]
    log(f"best relaxation objective={br['objective']:.6f} "
        f"(gap {relax['gap_to_optimum']:+.6f}), residual={br['constraint_residual']:+.4f}")
    log(f"rounding -> {br['rounding']['rounded_asset_indices']} "
        f"C0={br['rounding']['objective']:.6f} feasible={br['rounding']['is_feasible']}")

    # --- 2. classical baselines ---
    log("\n--- classical baselines ---")
    greedy = p3.greedy_forward_selection(inst)
    sa = p3.run_simulated_annealing(inst, SA_RESTARTS, SA_STEPS, SA_T0, SA_T1,
                                    base_seed=777)
    greedy["gap_to_optimum"] = greedy["objective"] - C_best
    sa["gap_to_optimum"] = sa["best_cost"] - C_best
    log(f"greedy   {greedy['asset_indices']} C0={greedy['objective']:.6f} "
        f"gap={greedy['gap_to_optimum']:+.2e} rt={greedy['runtime_s']*1e3:.2f}ms")
    log(f"SA best  {sa['best_asset_indices']} C0={sa['best_cost']:.6f} "
        f"gap={sa['gap_to_optimum']:+.2e} mean={sa['mean_best_cost']:.6f} "
        f"rt={sa['total_runtime_s']:.2f}s")
    classical_out = {
        "team": "axiom", "phase": 3, "study": "classical_baselines",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": {"sa": settings["sa"]},
        "greedy": greedy, "simulated_annealing": sa,
        "package_versions": versions,
    }
    _dump(os.path.join(OUT_DIR, "classical_baselines.json"), classical_out)

    # --- 3. Method A ---
    log("\n--- Method A: constraint-preserving XY-QAOA ---")
    method_a = exp.run_method_a(inst, bench, DEPTHS, N_STARTS, MAXITER,
                                BASE_SEED, SHOTS)
    method_a.update({"team": "axiom", "phase": 3,
                     "generated_utc": datetime.now(timezone.utc).isoformat(),
                     "instance": instance_block, "exact_reference": exact_block,
                     "settings": {"depths": list(DEPTHS), "n_starts": N_STARTS,
                                  "optimizer": q2.OPTIMIZER_NAME,
                                  "maxiter": MAXITER, "base_seed": BASE_SEED,
                                  "shots": SHOTS,
                                  "note": ("Constraint-preserving XY ring mixer; "
                                           "no penalty; exact subspace simulation "
                                           "replaces the full circuit in the loop")},
                     "package_versions": versions})
    _dump(os.path.join(OUT_DIR, "method_a_xy_qaoa.json"), method_a)
    for r in method_a["records"]:
        log(f"  p={r['p']} E*={r['best_energy_C0_exact']:.6f} "
            f"bestFeas={r['sampling']['sampled_metrics']['best_feasible_unpenalized_cost']:.6f} "
            f"P(opt)exact={r['sampling']['exact_statevector_metrics']['optimal_solution_probability']:.4f} "
            f"wall={r['optimization_wall_clock_s']:.2f}s")

    # --- 4. Method B ---
    log("\n--- Method B: warm-start QAOA ---")
    method_b = exp.run_method_b(inst, bench, relax["best_run"], DEPTHS, N_STARTS,
                                MAXITER, p3.WARM_START_SEED, LAMBDA, SHOTS)
    method_b.update({"team": "axiom", "phase": 3,
                     "generated_utc": datetime.now(timezone.utc).isoformat(),
                     "instance": instance_block, "exact_reference": exact_block,
                     "settings": {"depths": list(DEPTHS), "n_starts": N_STARTS,
                                  "optimizer": q2.OPTIMIZER_NAME,
                                  "maxiter": MAXITER, "lambda": LAMBDA,
                                  "base_seed": p3.WARM_START_SEED,
                                  "shots": SHOTS,
                                  "initial_state": "product Ry(2 arcsin sqrt(c*))"},
                     "package_versions": versions})
    _dump(os.path.join(OUT_DIR, "method_b_warm_start.json"), method_b)
    for r in method_b["records"]:
        em = r["sampling"]["exact_statevector_metrics"]
        log(f"  p={r['p']} E*={r['best_energy_C_lambda_exact']:.6f} "
            f"bestFeas={r['sampling']['sampled_metrics']['best_feasible_unpenalized_cost']:.6f} "
            f"exactFeas={em['feasibility_rate']:.4f} "
            f"P(opt)exact={em['optimal_solution_probability']:.4f} "
            f"wall={r['optimization_wall_clock_s']:.2f}s")

    # --- 5. Method C ---
    log("\n--- Method C: penalized standard QAOA ---")
    method_c = exp.run_method_c(inst, bench, DEPTHS, N_STARTS, MAXITER,
                                BASE_SEED, LAMBDA, SHOTS)
    method_c.update({"team": "axiom", "phase": 3,
                     "generated_utc": datetime.now(timezone.utc).isoformat(),
                     "instance": instance_block, "exact_reference": exact_block,
                     "settings": {"depths": list(DEPTHS), "n_starts": N_STARTS,
                                  "optimizer": q2.OPTIMIZER_NAME,
                                  "maxiter": MAXITER, "lambda": LAMBDA,
                                  "base_seed": BASE_SEED, "shots": SHOTS},
                     "package_versions": versions})
    _dump(os.path.join(OUT_DIR, "method_c_penalty_qaoa.json"), method_c)
    for r in method_c["records"]:
        em = r["sampling"]["exact_statevector_metrics"]
        log(f"  p={r['p']} E*={r['best_energy_C_lambda_exact']:.6f} "
            f"bestFeas={r['sampling']['sampled_metrics']['best_feasible_unpenalized_cost']:.6f} "
            f"exactFeas={em['feasibility_rate']:.4f} "
            f"P(opt)exact={em['optimal_solution_probability']:.4f} "
            f"wall={r['optimization_wall_clock_s']:.2f}s")

    # --- 6. penalty diagnostic ---
    log("\n--- penalty diagnostic (Method C, p=2) ---")
    penalty_diag = exp.run_penalty_diagnostic(inst, bench,
                                              p3.PRIMARY_PENALTY_DIAGNOSTIC, 2,
                                              N_STARTS, MAXITER, BASE_SEED, SHOTS)
    for r in penalty_diag["records"]:
        em = r["sampling"]["exact_statevector_metrics"]
        log(f"  lambda={r['lambda']}: exactFeas={em['feasibility_rate']:.4f} "
            f"P(opt)exact={em['optimal_solution_probability']:.4f} "
            f"E[C0]exact={em['expected_unpenalized_cost']:.6f}")

    # --- 7. comparison summary ---
    method_best = {
        "A": _compact(min(method_a["records"],
                          key=lambda r: (r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"]
                                         if r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"] is not None else 1e9))),
        "B": _compact(min(method_b["records"],
                          key=lambda r: (r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"]
                                         if r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"] is not None else 1e9))),
        "C": _compact(min(method_c["records"],
                          key=lambda r: (r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"]
                                         if r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"] is not None else 1e9))),
    }
    comparison = {
        "team": "axiom", "phase": 3, "study": "comparison_summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "instance": instance_block, "exact_reference": exact_block,
        "settings": settings,
        "continuous_relaxation": {
            "best_objective": br["objective"],
            "gap_to_optimum": relax["gap_to_optimum"],
            "constraint_residual": br["constraint_residual"],
            "rounded_portfolio": br["rounding"],
        },
        "classical": {
            "greedy": {"objective": greedy["objective"],
                       "asset_indices": greedy["asset_indices"],
                       "gap_to_optimum": greedy["gap_to_optimum"],
                       "runtime_s": greedy["runtime_s"]},
            "simulated_annealing": {
                "best_cost": sa["best_cost"],
                "best_asset_indices": sa["best_asset_indices"],
                "gap_to_optimum": sa["gap_to_optimum"],
                "mean_best_cost": sa["mean_best_cost"],
                "std_best_cost": sa["std_best_cost"],
                "total_runtime_s": sa["total_runtime_s"],
                "total_evaluations": sa["total_evaluations"]},
        },
        "quantum": {
            "A_xy_qaoa": {"records": [_compact(r) for r in method_a["records"]],
                          "best": method_best["A"]},
            "B_warm_start": {"records": [_compact(r) for r in method_b["records"]],
                             "best": method_best["B"]},
            "C_penalty_qaoa": {"records": [_compact(r) for r in method_c["records"]],
                               "best": method_best["C"]},
        },
        "penalty_diagnostic": [_compact(r) for r in penalty_diag["records"]],
        "verdict": {},
        "package_versions": versions,
    }
    opt_assets = set(bench["optimal_asset_indices"])
    verdict = {
        "exact_optimum_assets": sorted(opt_assets),
        "greedy_is_optimal": bool(set(greedy["asset_indices"]) == opt_assets),
        "sa_is_optimal": bool(set(sa["best_asset_indices"]) == opt_assets),
        "note": ("Classical greedy and feasible-swap SA both reach the exact "
                 "optimum on this instance, so no quantum advantage is claimed. "
                 "The quantum methods are compared on solution quality, "
                 "feasibility, optimal-solution probability and cost."),
    }
    for m in ("A", "B", "C"):
        verdict[f"method_{m}_best_feasible"] = method_best[m]["best_feasible_sampled"]
    comparison["verdict"] = verdict
    _dump(os.path.join(OUT_DIR, "comparison_summary.json"), comparison)

    # --- 8. figures ---
    log("\n--- figures ---")
    from phase2_experiments import _jsonable as _j
    figs = exp.make_figures(method_a, method_b, method_c,
                            {"greedy": greedy, "simulated_annealing": sa},
                            comparison, FIG_DIR)
    fdata = exp.figure_data(method_a, method_b, method_c,
                            {"greedy": greedy, "simulated_annealing": sa})
    _dump(os.path.join(OUT_DIR, "figure_data.json"), fdata)
    log(f"  {len(figs)} figures written")

    # --- 9. validation ---
    log("\n--- validation vs independent Qiskit circuits ---")
    inst6 = core.generate_instance(6, 3)
    t_val = time.perf_counter()
    val_a = _validate_method_a(inst, depths=(1, 3))
    val_b = _validate_full_space(inst6, "B")
    val_c = _validate_full_space(inst6, "C")
    val_time = time.perf_counter() - t_val
    for v in val_a:
        log(f"  MethodA p={v['p']} fid={v['fidelity_subspace_vs_qiskit']:.12f} "
            f"dE={v['abs_energy_diff']:.2e} infeasMass={v['infeasible_mass_qiskit']:.2e}")
    log(f"  MethodB n=6 minFid={min(v['fidelity_vs_qiskit'] for v in val_b):.12f}")
    log(f"  MethodC n=6 minFid={min(v['fidelity_vs_qiskit'] for v in val_c):.12f}")

    test_stats, tests_failed = _run_tests()

    validation = {
        "team": "axiom", "phase": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "package_versions": versions,
        "tests": test_stats,
        "method_a_subspace_vs_qiskit_n12": val_a,
        "method_b_full_vs_qiskit_n6": val_b,
        "method_c_full_vs_qiskit_n6": val_c,
        "exact_reference": exact_block,
        "continuous_relaxation_check": {
            "relaxation_objective": br["objective"],
            "C_best": C_best,
            "relaxation_is_lower_bound": bool(br["objective"] <= C_best + 1e-9),
            "gap": relax["gap_to_optimum"],
        },
        "sampling_note": ("Sampled results are draws from the exact statevector "
                          "distribution via a seeded multinomial; they are NOT "
                          "hardware measurements. Exact-statevector metrics are "
                          "reported alongside and kept separate."),
        "metric_definitions": {
            "C_best": "exact optimal feasible unpenalized cost",
            "C_worst": "exact worst feasible unpenalized cost",
            "feasible_normalized_score":
                "(C_worst - E[C0 | feasible]) / (C_worst - C_best); null if no "
                "feasible mass or C_worst == C_best",
            "optimal_solution_probability":
                "probability mass on any globally optimal feasible portfolio",
            "gap_to_optimum": "objective - C_best (0 is optimal)",
        },
        "known_limitations": [
            "Single synthetic 12-asset instance; exhaustive search is the true reference.",
            "Classical greedy already attains the exact optimum; SA matches it, so no "
            "quantum advantage is demonstrated.",
            "Sampled metrics come from exact-statevector multinomial draws, not hardware.",
            "QAOA parameters are optimised with COBYLA; Method A uses an exact "
            "subspace simulation in the loop and a Qiskit circuit only for validation.",
            "Circuit depths reported are logical (StatePreparation and "
            "PauliEvolutionGate are opaque), not transpiled depths.",
        ],
        "validation_wall_clock_s": float(val_time),
    }
    _dump(os.path.join(OUT_DIR, "validation_report.json"), validation)

    total = time.perf_counter() - t_start
    log(f"\ntests: {test_stats}")
    log(f"total wall clock: {total:.1f}s")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "phase3-run.log"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    return 1 if tests_failed else 0


if __name__ == "__main__":
    sys.exit(main())
