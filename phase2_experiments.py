"""Phase 2 experiments: depth study, penalty study, and plotting.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

This module contains only experiment orchestration and figure generation. All
core QAOA construction, optimization, sampling and metric definitions live in
:mod:`phase2_qaoa`; all model definitions live in :mod:`phase1_core`.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Sequence

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa

SAMPLE_SEED_BASE = 90000
INIT_RANGE = {"gamma": [0.0, 2.0 * math.pi], "beta": [0.0, math.pi]}
SAMPLING_METHOD = ("exact-statevector multinomial draw (numpy default_rng); "
                    "NOT hardware measurement")


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def _depth_sample_seed(p: int) -> int:
    return SAMPLE_SEED_BASE + 100 * p


def _penalty_sample_seed(lam: float) -> int:
    return SAMPLE_SEED_BASE + 2000 + int(round(lam * 1000))


def _start_seed(kind: str, key: float, base_seed: int) -> int:
    if kind == "depth":
        return base_seed + 7 * int(key)
    return base_seed + 7000 + 7 * int(round(key * 1000))


def _summarise_starts(starts: List[qaoa.StartResult]) -> Dict:
    successful = [s for s in starts if s.success]
    valid = [s for s in starts if np.isfinite(s.energy_final)]
    energies = [s.energy_final for s in successful] if successful else []
    nfevs = sorted(s.nfev for s in starts)
    median_nfev = (nfevs[len(nfevs) // 2] if len(nfevs) % 2 == 1
                   else 0.5 * (nfevs[len(nfevs) // 2 - 1] + nfevs[len(nfevs) // 2]))
    return {
        "num_starts": len(starts),
        "num_valid_finite": len(valid),
        "num_successful": len(successful),
        "fraction_successful": (len(successful) / len(starts)) if starts else 0.0,
        "energy_mean_successful": (float(np.mean(energies)) if energies else None),
        "energy_std_successful": (float(np.std(energies)) if energies else None),
        "best_nfev": int(min(nfevs)) if nfevs else None,
        "median_nfev": float(median_nfev) if nfevs else None,
    }


def _sample_and_score(sim: qaoa.FastQAOASimulator, theta: Sequence[float],
                      ctx: qaoa.MetricContext, seed: int) -> Dict:
    probs = sim.probabilities(theta)
    counts = qaoa.sample_counts_exact(probs, qaoa.SHOTS, seed)
    assert sum(counts.values()) == qaoa.SHOTS, "shot counts must sum to shots"
    return {
        "sampling_method": SAMPLING_METHOD,
        "shots": qaoa.SHOTS,
        "sample_seed": int(seed),
        "counts_checked_sum": int(sum(counts.values())),
        "sampled_metrics": qaoa.metrics_from_counts(counts, ctx),
        "exact_statevector_metrics": qaoa.metrics_from_probs(probs, ctx),
    }


def run_depth_study(inst: core.PortfolioInstance, depths: Sequence[int],
                    lam: float, n_starts: int, base_seed: int,
                    maxiter: int, benchmark: Dict) -> Dict:
    """Phase 2A: multistart QAOA over circuit depths at fixed penalty."""
    n = inst.n
    cost_op = qaoa.cost_operator(inst, lam)
    diag = core.operator_diagonal(cost_op)
    records = []
    for p in depths:
        sim = qaoa.FastQAOASimulator(n, p, diag)
        t0 = time.perf_counter()
        starts, best = qaoa.multistart_optimize(
            sim, n_starts, _start_seed("depth", p, base_seed),
            maxiter=maxiter)
        wall = time.perf_counter() - t0
        ctx = qaoa.MetricContext.build(inst, lam, benchmark)
        sampling = _sample_and_score(sim, best.x_final, ctx,
                                     _depth_sample_seed(p))
        record = {
            "p": int(p),
            "lambda": float(lam),
            "n_qubits": n,
            "n_parameters": 2 * p,
            "optimizer": qaoa.OPTIMIZER_NAME,
            "optimizer_options": {"maxiter": int(maxiter),
                                  "rhobeg": qaoa.DEFAULT_RHOBEG},
            "initialisation_range": INIT_RANGE,
            "best_energy": float(best.energy_final),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "best_start_nfev": int(best.nfev),
            "starts_summary": _summarise_starts(starts),
            "starts": [s.__dict__ for s in starts],
            "total_wall_clock_s": float(wall),
            "sampling": sampling,
        }
        records.append(_jsonable(record))
    return {"study": "depth", "lambda": float(lam), "depths": list(depths),
            "n_starts": int(n_starts), "records": records}


def run_penalty_study(inst: core.PortfolioInstance, penalties: Sequence[float],
                      p: int, n_starts: int, base_seed: int, maxiter: int,
                      benchmark: Dict) -> Dict:
    """Phase 2B: multistart QAOA over penalty coefficients at fixed depth."""
    n = inst.n
    records = []
    for lam in penalties:
        cost_op = qaoa.cost_operator(inst, lam)
        diag = core.operator_diagonal(cost_op)
        sim = qaoa.FastQAOASimulator(n, p, diag)
        t0 = time.perf_counter()
        starts, best = qaoa.multistart_optimize(
            sim, n_starts, _start_seed("penalty", lam, base_seed),
            maxiter=maxiter)
        wall = time.perf_counter() - t0
        ctx = qaoa.MetricContext.build(inst, lam, benchmark)
        sampling = _sample_and_score(sim, best.x_final, ctx,
                                     _penalty_sample_seed(lam))
        record = {
            "lambda": float(lam),
            "p": int(p),
            "n_qubits": n,
            "n_parameters": 2 * p,
            "optimizer": qaoa.OPTIMIZER_NAME,
            "optimizer_options": {"maxiter": int(maxiter),
                                  "rhobeg": qaoa.DEFAULT_RHOBEG},
            "initialisation_range": INIT_RANGE,
            "best_penalized_energy": float(best.energy_final),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "best_start_nfev": int(best.nfev),
            "starts_summary": _summarise_starts(starts),
            "starts": [s.__dict__ for s in starts],
            "total_wall_clock_s": float(wall),
            "sampling": sampling,
        }
        records.append(_jsonable(record))
    return {"study": "penalty", "p": int(p), "penalties": list(penalties),
            "n_starts": int(n_starts), "records": records}


def random_feasible_baseline(inst: core.PortfolioInstance, benchmark: Dict,
                             n_draws: int, seed: int) -> Dict:
    """Optional benchmark: mean/worst/best C0 over uniformly random feasible x."""
    rng = np.random.default_rng(seed)
    feasible = core.feasible_assignments(inst.n, inst.K)
    idx = rng.integers(0, feasible.shape[0], size=n_draws)
    vals = np.array([core.cost_unpenalized(feasible[i], inst.mu, inst.Sigma, inst.q)
                     for i in idx])
    return {
        "method": "uniform draws over the feasible set",
        "seed": int(seed),
        "n_draws": int(n_draws),
        "mean_unpenalized_cost": float(np.mean(vals)),
        "best_unpenalized_cost": float(np.min(vals)),
        "worst_unpenalized_cost": float(np.max(vals)),
        "C_best": float(benchmark["best_feasible_objective"]),
        "C_worst": float(benchmark["worst_feasible_objective"]),
    }


def make_figures(depth: Dict, penalty: Dict, exact_penalty: Dict,
                 fig_dir: str) -> List[str]:
    """Write all Phase 2 figures; return the list of created file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    created: List[str] = []

    # --- depth: normalised feasible score vs p (sampled and exact) ---
    ps = [r["p"] for r in depth["records"]]
    samp = [r["sampling"]["sampled_metrics"]["feasible_normalized_score"]
            for r in depth["records"]]
    exact = [r["sampling"]["exact_statevector_metrics"]["feasible_normalized_score"]
             for r in depth["records"]]

    def _nan(v):
        return np.nan if v is None else v

    fig, ax = plt.subplots()
    ax.plot(ps, [_nan(v) for v in samp], "o-", label="sampled (4096 shots)")
    ax.plot(ps, [_nan(v) for v in exact], "s--", label="exact statevector")
    ax.set_xlabel("depth p")
    ax.set_ylabel("normalised feasible score")
    ax.set_title("Beginner QAOA: normalised feasible score vs depth")
    ax.legend()
    path = os.path.join(fig_dir, "depth_score_vs_p.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # --- depth: best penalized energy vs p ---
    fig, ax = plt.subplots()
    ax.plot(ps, [r["best_energy"] for r in depth["records"]], "o-",
            color="tab:red")
    ax.set_xlabel("depth p")
    ax.set_ylabel("best penalized expectation <H_C>")
    ax.set_title("Beginner QAOA: best penalized energy vs depth")
    path = os.path.join(fig_dir, "depth_energy_vs_p.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # --- penalty plots ---
    lams = [r["lambda"] for r in penalty["records"]]
    fig, ax = plt.subplots()
    ax.plot(lams, [r["sampling"]["sampled_metrics"]["feasibility_rate"]
                   for r in penalty["records"]], "o-", label="sampled")
    ax.plot(lams, [r["sampling"]["exact_statevector_metrics"]["feasibility_rate"]
                   for r in penalty["records"]], "s--", label="exact")
    ax.axvline(exact_penalty["penalty_threshold"], color="gray", ls=":",
               label="exact threshold")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("feasibility rate")
    ax.set_title("Beginner QAOA: feasibility rate vs penalty")
    ax.legend()
    path = os.path.join(fig_dir, "penalty_feasibility_vs_lambda.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots()
    ax.plot(lams, [r["sampling"]["sampled_metrics"]["optimal_solution_probability"]
                   for r in penalty["records"]], "o-", label="sampled")
    ax.plot(lams, [r["sampling"]["exact_statevector_metrics"]["optimal_solution_probability"]
                   for r in penalty["records"]], "s--", label="exact")
    ax.axvline(exact_penalty["penalty_threshold"], color="gray", ls=":",
               label="exact threshold")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("P(optimal feasible)")
    ax.set_title("Beginner QAOA: optimal-solution probability vs penalty")
    ax.legend()
    path = os.path.join(fig_dir, "penalty_optimalprob_vs_lambda.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots()
    ax.plot(lams, [_nan(r["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"])
                   for r in penalty["records"]], "o-", label="sampled best")
    ax.axhline(exact_penalty["best_feasible_unpenalized_cost"], color="green",
               ls="--", label="exact optimum C_best")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("best feasible C0")
    ax.set_title("Beginner QAOA: best feasible sampled cost vs penalty")
    ax.legend()
    path = os.path.join(fig_dir, "penalty_bestcost_vs_lambda.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots()
    feas = [1.0 if exact_penalty["per_lambda"][f"{l:g}"]["global_minimizer_feasible"]
            else 0.0 for l in lams]
    ax.step(lams, feas, where="mid", marker="o")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("global minimizer feasible (1/0)")
    ax.set_title("Exact ground-state feasibility vs penalty")
    path = os.path.join(fig_dir, "exact_groundstate_feasible_vs_lambda.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    return created
