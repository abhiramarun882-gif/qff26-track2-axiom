"""Phase 3 experiments: orchestration, metrics, and figures.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

Runs the three quantum methods (A: XY-QAOA, B: warm-start QAOA, C: penalized
standard QAOA), the two classical baselines, and assembles the comparison.
All heavy lifting lives in :mod:`phase3_methods` and :mod:`phase2_qaoa`.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import phase1_core as core
import phase2_qaoa as q2
import phase3_methods as p3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expected_unpenalized(weights: Dict[int, float], total: float,
                         ctx: q2.MetricContext) -> float:
    s = 0.0
    for idx, w in weights.items():
        x = core.index_to_assets(int(idx), ctx.n)
        s += w * core.cost_unpenalized(x, ctx.mu, ctx.Sigma, ctx.q)
    return float(s / total)


def _distribution_metrics(weights: Dict[int, float], total: float,
                          ctx: q2.MetricContext) -> Dict:
    m = q2.aggregate_metrics(weights, total, ctx)
    m["expected_unpenalized_cost"] = expected_unpenalized(weights, total, ctx)
    return m


def _sample_block(probs_full: np.ndarray, ctx: q2.MetricContext, shots: int,
                  seed: int) -> Dict:
    counts = q2.sample_counts_exact(probs_full, shots, seed)
    if sum(counts.values()) != shots:
        raise RuntimeError("shot counts do not sum to shots")
    idx = np.nonzero(probs_full > 0)[0]
    exact = _distribution_metrics({int(i): float(probs_full[i]) for i in idx},
                                  float(probs_full.sum()), ctx)
    sampled = _distribution_metrics({int(k): float(v) for k, v in counts.items()},
                                    float(shots), ctx)
    return {
        "sampling_method": ("exact-statevector multinomial draw (numpy "
                            "default_rng); NOT hardware measurement"),
        "shots": int(shots),
        "sample_seed": int(seed),
        "counts_checked_sum": int(sum(counts.values())),
        "sampled_metrics": sampled,
        "exact_statevector_metrics": exact,
    }


def _space_probs_from_subspace(sim: p3.WeightSubspaceSimulator,
                               theta: Sequence[float], n: int) -> np.ndarray:
    probs = np.zeros(1 << n, dtype=float)
    for idx, val in sim.prob_dict(theta).items():
        probs[idx] = val
    return probs


def _starts_summary(starts: List[q2.StartResult]) -> Dict:
    successful = [s for s in starts if s.success]
    valid = [s for s in starts if np.isfinite(s.energy_final)]
    energies = [s.energy_final for s in successful] if successful else []
    nfevs = sorted(s.nfev for s in starts)
    median = (nfevs[len(nfevs) // 2] if len(nfevs) % 2
              else 0.5 * (nfevs[len(nfevs) // 2 - 1] + nfevs[len(nfevs) // 2]))
    return {
        "num_starts": len(starts),
        "num_valid_finite": len(valid),
        "num_successful": len(successful),
        "fraction_successful": (len(successful) / len(starts)) if starts else 0.0,
        "energy_mean_successful": float(np.mean(energies)) if energies else None,
        "energy_std_successful": float(np.std(energies)) if energies else None,
        "energy_min": float(min(s.energy_final for s in valid)) if valid else None,
        "energy_max": float(max(s.energy_final for s in valid)) if valid else None,
        "best_nfev": int(min(nfevs)) if nfevs else None,
        "median_nfev": float(median) if nfevs else None,
    }


def _warmstart_theta(prev_theta: Sequence[float], reps: int) -> np.ndarray:
    """Extend a depth-(reps-1) theta to depth reps by repeating the last angle."""
    prev = np.asarray(prev_theta, dtype=float)
    pm = reps - 1
    g = prev[:pm].tolist() + [float(prev[pm - 1])]
    b = prev[pm:].tolist() + [float(prev[-1])]
    return np.array(g + b, dtype=float)


def _circuit_resources(inst: core.PortfolioInstance, reps: int) -> Dict:
    """Logical resource report for Method A's Qiskit reference circuit.

    ``StatePreparation`` and each ``PauliEvolutionGate`` are counted as single
    opaque instructions at this logical level; a decomposition estimate is
    provided separately and clearly labelled as approximate.
    """
    cost_op0 = core.qubo_to_pauli(inst.mu, inst.Sigma, inst.q, inst.K, 0.0)[0]
    qc = p3.build_method_a_circuit(inst, cost_op0, reps)
    return {
        "num_qubits": inst.n,
        "reps": reps,
        "num_trainable_parameters": 2 * reps,
        "logical_depth_opaque": int(qc.circuit.depth()),
        "logical_size_opaque": int(qc.circuit.size()),
        "state_preparation_included": True,
        "state_preparation_is_opaque": True,
        "two_qubit_gate_count": None,
        "two_qubit_gate_count_note": (
            "Not reliably available at the logical level because "
            "StatePreparation and PauliEvolutionGate are opaque. A standard "
            "decomposition of the XY ring mixer contributes 2*n two-qubit "
            "gates per layer (n XX + n YY); the ZZ cost layer and the state "
            "preparation add further, unspecified cost."),
        "depth_is_logical_not_transpiled": True,
    }


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
def run_method_a(inst: core.PortfolioInstance, benchmark: Dict,
                 depths: Sequence[int] = p3.DEPTHS, n_starts: int = p3.N_STARTS,
                 maxiter: int = p3.MAXITER, base_seed: int = core.DEFAULT_SEED,
                 shots: int = p3.SHOTS, include_warmstart: bool = True) -> Dict:
    """Constraint-preserving XY-QAOA on the objective-only Hamiltonian."""
    n, K = inst.n, inst.K
    cost_op0 = core.qubo_to_pauli(inst.mu, inst.Sigma, inst.q, K, 0.0)[0]
    diag0 = core.operator_diagonal(cost_op0)
    ctx = q2.MetricContext.build(inst, 0.0, benchmark)
    records = []
    prev_best: Optional[q2.StartResult] = None
    for p in depths:
        sim = p3.WeightSubspaceSimulator(n, K, diag0, p)
        seed = base_seed + 11 * p
        t0 = time.perf_counter()
        starts, best = q2.multistart_optimize(sim, n_starts, seed, maxiter)
        all_starts = list(starts)
        if include_warmstart and prev_best is not None:
            x0 = _warmstart_theta(prev_best.x_final, p)
            all_starts.append(q2.optimize_single_start(sim, x0, len(starts),
                                                       maxiter))
            valid = [s for s in all_starts if np.isfinite(s.energy_final)]
            best = min(valid, key=lambda s: s.energy_final)
        wall = time.perf_counter() - t0
        probs = _space_probs_from_subspace(sim, best.x_final, n)
        sampling = _sample_block(probs, ctx, shots, base_seed + 1000 + p)
        records.append(_jsonable({
            "method": "A_xy_qaoa",
            "p": int(p),
            "n_qubits": n,
            "n_parameters": 2 * p,
            "lambda": None,
            "optimizer": q2.OPTIMIZER_NAME,
            "optimizer_options": {"maxiter": int(maxiter),
                                  "rhobeg": q2.DEFAULT_RHOBEG},
            "initialisation": ("random gammas U[0,2pi), betas U[0,pi); "
                               "warm-started starts flagged separately"),
            "best_energy_C0_exact": float(sim.energy(best.x_final)),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "best_is_warmstart": bool(best.start_index >= n_starts),
            "starts": [s.__dict__ for s in all_starts],
            "starts_summary": _starts_summary(all_starts),
            "sampling": sampling,
            "circuit_resources": _circuit_resources(inst, p),
            "optimization_wall_clock_s": float(wall),
        }))
        prev_best = best
    return {"method": "A_xy_qaoa", "depths": list(depths),
            "n_starts": int(n_starts), "records": records}


def run_method_b(inst: core.PortfolioInstance, benchmark: Dict,
                 relaxation: Dict, depths: Sequence[int] = p3.DEPTHS,
                 n_starts: int = p3.N_STARTS, maxiter: int = p3.MAXITER,
                 base_seed: int = p3.WARM_START_SEED, lam: float = p3.LAMBDA,
                 shots: int = p3.SHOTS) -> Dict:
    """Warm-start QAOA: product state from relaxation + penalized X-mixer QAOA."""
    n = inst.n
    cost_op = q2.cost_operator(inst, lam)
    diag = core.operator_diagonal(cost_op)
    init_amp = p3.product_state_amplitudes(relaxation["final_point"])
    ctx = q2.MetricContext.build(inst, lam, benchmark)
    records = []
    for p in depths:
        sim = p3.FullSpaceSimulator(n, p, diag, init_amp)
        seed = base_seed + 11 * p
        t0 = time.perf_counter()
        starts, best = q2.multistart_optimize(sim, n_starts, seed, maxiter)
        wall = time.perf_counter() - t0
        probs = sim.probabilities(best.x_final)
        sampling = _sample_block(probs, ctx, shots, base_seed + 1000 + p)
        records.append(_jsonable({
            "method": "B_warm_start",
            "p": int(p),
            "n_qubits": n,
            "n_parameters": 2 * p,
            "lambda": float(lam),
            "initial_state": "product Ry(2 arcsin sqrt(c*))",
            "mixer": "standard X (sum_i X_i)",
            "optimizer": q2.OPTIMIZER_NAME,
            "optimizer_options": {"maxiter": int(maxiter),
                                  "rhobeg": q2.DEFAULT_RHOBEG},
            "best_energy_C_lambda_exact": float(sim.energy(best.x_final)),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "starts": [s.__dict__ for s in starts],
            "starts_summary": _starts_summary(starts),
            "sampling": sampling,
            "optimization_wall_clock_s": float(wall),
        }))
    return {"method": "B_warm_start", "depths": list(depths),
            "n_starts": int(n_starts), "relaxation": relaxation,
            "records": records}


def run_method_c(inst: core.PortfolioInstance, benchmark: Dict,
                 depths: Sequence[int] = p3.DEPTHS, n_starts: int = p3.N_STARTS,
                 maxiter: int = p3.MAXITER, base_seed: int = core.DEFAULT_SEED,
                 lam: float = p3.LAMBDA, shots: int = p3.SHOTS) -> Dict:
    """Penalized standard QAOA (|+>^n, X mixer, penalized H_C)."""
    n = inst.n
    cost_op = q2.cost_operator(inst, lam)
    diag = core.operator_diagonal(cost_op)
    ctx = q2.MetricContext.build(inst, lam, benchmark)
    records = []
    for p in depths:
        sim = p3.FullSpaceSimulator(n, p, diag, p3.uniform_plus_amplitudes(n))
        seed = base_seed + 11 * p
        t0 = time.perf_counter()
        starts, best = q2.multistart_optimize(sim, n_starts, seed, maxiter)
        wall = time.perf_counter() - t0
        probs = sim.probabilities(best.x_final)
        sampling = _sample_block(probs, ctx, shots, base_seed + 1000 + p)
        records.append(_jsonable({
            "method": "C_penalty_qaoa",
            "p": int(p),
            "n_qubits": n,
            "n_parameters": 2 * p,
            "lambda": float(lam),
            "initial_state": "|+>^n",
            "mixer": "standard X (sum_i X_i)",
            "optimizer": q2.OPTIMIZER_NAME,
            "optimizer_options": {"maxiter": int(maxiter),
                                  "rhobeg": q2.DEFAULT_RHOBEG},
            "best_energy_C_lambda_exact": float(sim.energy(best.x_final)),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "starts": [s.__dict__ for s in starts],
            "starts_summary": _starts_summary(starts),
            "sampling": sampling,
            "optimization_wall_clock_s": float(wall),
        }))
    return {"method": "C_penalty_qaoa", "depths": list(depths),
            "n_starts": int(n_starts), "lambda": float(lam), "records": records}


def run_penalty_diagnostic(inst: core.PortfolioInstance, benchmark: Dict,
                           penalties: Sequence[float] = p3.PRIMARY_PENALTY_DIAGNOSTIC,
                           p: int = 2, n_starts: int = p3.N_STARTS,
                           maxiter: int = p3.MAXITER,
                           base_seed: int = core.DEFAULT_SEED,
                           shots: int = p3.SHOTS) -> Dict:
    n = inst.n
    records = []
    for lam in penalties:
        cost_op = q2.cost_operator(inst, lam)
        diag = core.operator_diagonal(cost_op)
        ctx = q2.MetricContext.build(inst, lam, benchmark)
        sim = p3.FullSpaceSimulator(n, p, diag, p3.uniform_plus_amplitudes(n))
        seed = base_seed + 5000 + int(round(lam * 1000))
        t0 = time.perf_counter()
        starts, best = q2.multistart_optimize(sim, n_starts, seed, maxiter)
        wall = time.perf_counter() - t0
        probs = sim.probabilities(best.x_final)
        sampling = _sample_block(probs, ctx, shots, seed + 1)
        records.append(_jsonable({
            "method": "C_penalty_qaoa", "p": int(p), "lambda": float(lam),
            "best_energy_C_lambda_exact": float(sim.energy(best.x_final)),
            "best_theta": [float(v) for v in best.x_final],
            "best_start_index": int(best.start_index),
            "starts": [s.__dict__ for s in starts],
            "starts_summary": _starts_summary(starts),
            "sampling": sampling,
            "optimization_wall_clock_s": float(wall),
        }))
    return {"method": "C_penalty_qaoa_penalty_diagnostic", "p": int(p),
            "penalties": list(penalties), "n_starts": int(n_starts),
            "records": records}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _best_feasible(rec: Dict) -> Optional[float]:
    return rec["sampling"]["sampled_metrics"]["best_feasible_unpenalized_cost"]


def _opt_prob(rec: Dict, kind: str = "sampled") -> float:
    m = rec["sampling"][f"{kind}_metrics"]
    return m["optimal_solution_probability"]


def make_figures(method_a: Dict, method_b: Dict, method_c: Dict,
                 classical: Dict, comparison: Dict, fig_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    created: List[str] = []

    def save(fig, name):
        path = os.path.join(fig_dir, name)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    def nan(v):
        return np.nan if v is None else v

    C_best = comparison["exact_reference"]["C_best"]
    C_worst = comparison["exact_reference"]["C_worst"]

    methods = [("A", method_a), ("B", method_b), ("C", method_c)]
    colors = {"A": "tab:green", "B": "tab:blue", "C": "tab:red"}

    # 1. best feasible objective vs method and depth
    fig, ax = plt.subplots()
    for name, m in methods:
        ps = [r["p"] for r in m["records"]]
        vals = [nan(_best_feasible(r)) for r in m["records"]]
        ax.plot(ps, vals, "o-", color=colors[name], label=f"Method {name}")
    ax.axhline(C_best, color="black", ls="--", label="exact optimum")
    ax.axhline(classical["greedy"]["objective"], color="gray", ls=":",
               label="greedy")
    ax.axhline(classical["simulated_annealing"]["best_cost"], color="orange",
               ls="-.", label="SA best")
    ax.set_xticks([r["p"] for r in method_a["records"]])
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("best feasible sampled C0")
    ax.set_title("Best feasible objective by method and depth (Intermediate n=12)")
    ax.legend(fontsize=8)
    save(fig, "best_feasible_vs_method_depth.png")

    # 2. optimal-solution probability per method
    fig, ax = plt.subplots()
    for name, m in methods:
        ps = [r["p"] for r in m["records"]]
        ax.plot(ps, [_opt_prob(r, "exact_statevector") for r in m["records"]],
                "o-", color=colors[name], label=f"Method {name} (exact)")
        ax.plot(ps, [_opt_prob(r, "sampled") for r in m["records"]],
                "s--", color=colors[name], alpha=0.5)
    ax.set_xticks([r["p"] for r in method_a["records"]])
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("P(optimal feasible portfolio)")
    ax.set_title("Optimal-solution probability (solid=exact, dashed=sampled)")
    ax.legend(fontsize=8)
    save(fig, "optimal_prob_vs_depth.png")

    # 3. feasibility rates
    fig, ax = plt.subplots()
    for name, m in methods:
        ps = [r["p"] for r in m["records"]]
        vals = [r["sampling"]["exact_statevector_metrics"]["feasibility_rate"]
                for r in m["records"]]
        ax.plot(ps, vals, "o-", color=colors[name], label=f"Method {name}")
    ax.set_xticks([r["p"] for r in method_a["records"]])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel("feasibility rate (exact statevector)")
    ax.set_title("Feasibility rate: constraint-preserving (A) vs penalized (B, C)")
    ax.legend(fontsize=8)
    save(fig, "feasibility_vs_depth.png")

    # 4. objective distributions across optimizer starts (best C0 proxy = energy)
    fig, ax = plt.subplots()
    positions, data, labels = [], [], []
    pos = 1
    ticks = []
    for name, m in methods:
        for r in m["records"]:
            energies = [s["energy_final"] for s in r["starts"]
                        if np.isfinite(s["energy_final"])]
            data.append(energies)
            positions.append(pos)
            ticks.append(f"{name}{r['p']}")
            pos += 1
        pos += 0.5
    ax.boxplot(data, positions=positions, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylabel("final objective per start (method-native)")
    ax.set_title("Spread of final objective across optimizer starts")
    save(fig, "start_objective_distributions.png")

    # 5. SA distribution across restarts
    fig, ax = plt.subplots()
    sa = classical["simulated_annealing"]
    best = [r["best_cost"] for r in sa["restarts"]]
    final = [r["final_cost"] for r in sa["restarts"]]
    ax.hist(best, bins=min(20, max(1, len(set(best)))), alpha=0.7,
            label="best-so-far")
    ax.hist(final, bins=min(20, max(1, len(set(final)))), alpha=0.4,
            label="final")
    ax.axvline(C_best, color="black", ls="--", label="exact optimum")
    ax.set_xlabel("C0")
    ax.set_ylabel("count")
    ax.set_title(f"Simulated annealing over {sa['n_restarts']} restarts")
    ax.legend(fontsize=8)
    save(fig, "sa_objective_distribution.png")

    # 6. runtime vs best objective
    fig, ax = plt.subplots()
    for name, m in methods:
        for r in m["records"]:
            ax.scatter(r["optimization_wall_clock_s"], nan(_best_feasible(r)),
                       color=colors[name], marker="o",
                       label=f"Method {name}" if r["p"] == m["records"][0]["p"] else None)
    ax.axhline(C_best, color="black", ls="--", label="exact optimum")
    ax.scatter(classical["greedy"]["runtime_s"], classical["greedy"]["objective"],
               color="gray", marker="*", s=140, label="greedy")
    ax.scatter(sa["total_runtime_s"], sa["best_cost"], color="orange",
               marker="*", s=140, label="SA")
    ax.set_xlabel("optimization runtime (s)")
    ax.set_ylabel("best feasible C0")
    ax.set_title("Runtime vs best objective")
    ax.legend(fontsize=8)
    save(fig, "runtime_vs_objective.png")

    # 7. sampling distribution for best run of each method (feasible mass)
    fig, ax = plt.subplots()
    labels, feas, opt = [], [], []
    for name, m in methods:
        best_rec = max(m["records"], key=lambda r: _opt_prob(r, "exact_statevector"))
        em = best_rec["sampling"]["exact_statevector_metrics"]
        labels.append(f"{name} p{best_rec['p']}")
        feas.append(em["feasibility_rate"])
        opt.append(em["optimal_solution_probability"])
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, feas, width=0.4, label="feasible mass")
    ax.bar(xpos + 0.2, opt, width=0.4, label="optimal mass")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("probability mass (exact)")
    ax.set_title("Feasible vs optimal probability mass (best run per method)")
    ax.legend(fontsize=8)
    save(fig, "sampling_mass_best_runs.png")

    return created


def figure_data(method_a: Dict, method_b: Dict, method_c: Dict,
                classical: Dict) -> Dict:
    """Numeric data underlying the figures (JSON-serialisable)."""
    out = {"methods": {}}
    for name, m in (("A", method_a), ("B", method_b), ("C", method_c)):
        out["methods"][name] = [
            {
                "p": r["p"],
                "best_feasible_sampled": _best_feasible(r),
                "exact_optimal_probability": _opt_prob(r, "exact_statevector"),
                "sampled_optimal_probability": _opt_prob(r, "sampled"),
                "exact_feasibility_rate":
                    r["sampling"]["exact_statevector_metrics"]["feasibility_rate"],
                "sampled_feasibility_rate":
                    r["sampling"]["sampled_metrics"]["feasibility_rate"],
                "optimization_wall_clock_s": r["optimization_wall_clock_s"],
                "start_energies": [s["energy_final"] for s in r["starts"]],
            }
            for r in m["records"]
        ]
    out["sa_best_costs"] = [r["best_cost"] for r in
                            classical["simulated_annealing"]["restarts"]]
    out["sa_final_costs"] = [r["final_cost"] for r in
                             classical["simulated_annealing"]["restarts"]]
    return out


def _jsonable(obj):
    from phase2_experiments import _jsonable as j
    return j(obj)
