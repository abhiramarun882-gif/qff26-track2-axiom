"""Phase 4: hardware awareness and noisy simulation for Method A.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

Reuses the Phase 3 Method A circuit (`phase3_methods.build_method_a_circuit`)
and the Phase 1 objective/metrics, and evaluates two things:

* **Experiment A** - logical vs hardware-transpiled resources for the Method A
  circuit on ``FakeBrisbane`` (simulator only, no QPU jobs).
* **Experiment B** - ideal vs FakeBrisbane-noisy sampling of that same circuit.

Selection rule, decoding convention, and metric definitions are inherited from
Phases 1-3 and are re-stated here so the module is self-describing:

* Method A circuit = `StatePreparation(uniform_weight_state)` + ``p`` rounds of
  (objective-only cost evolution, periodic XY ring mixer), bound by
  `phase2_qaoa.bind_theta` with ``theta = [gamma_0..gamma_{p-1},
  beta_0..beta_{p-1}]``.
* Project index convention (``phase1_core``): ``x[i] = (index >> i) & 1``,
  i.e. asset ``i`` <-> qubit ``i`` with qubit 0 the least significant bit.
  A Qiskit/Aer count key is written with the *highest* qubit leftmost, so the
  correct decode is ``int(key, 2)`` - **not** a reversed string.
* Metrics come from `phase2_qaoa.aggregate_metrics` (Phase 3 definitions).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

import phase1_core as core
import phase2_qaoa as q2
import phase3_methods as p3

# ---------------------------------------------------------------------------
# Deterministic configuration
# ---------------------------------------------------------------------------
N = 12
K = 6
METHOD = "A"
REPS = 3
DECOMPOSE_REPS = 4
OPT_LEVEL = 3
SEED_TRANSPILER = 2026
SHOTS_IDEAL = 4096
SHOTS_NOISY = 1024
SEED_IDEAL = 2026
SEED_NOISY = 2026
NOISY_TIMEOUT_S = 1800.0
METHOD_A_ARTIFACT = os.path.join("outputs", "phase3", "method_a_xy_qaoa.json")

# Selection criterion for the Method A run used throughout Phase 4. Identical
# to the Phase 3 follow-up rule: the record with the highest exact
# optimal-solution probability, ties broken by lower energy then smaller p.
REPRESENTATIVE_CRITERION = (
    "max exact optimal-solution probability, ties broken by lower "
    "best_energy_C0_exact then smaller p (same rule as the Phase 3 "
    "final follow-up)")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Method A circuit selection + provenance
# ---------------------------------------------------------------------------
def load_method_a(project_root: Optional[str] = None) -> Dict:
    """Load the selected Method A run and its provenance.

    Returns a dict with the instance, the cost operator, the bound logical
    circuit, the parameter vector and full provenance. Raises if the artifact
    is missing or does not contain an unambiguous ``p=REPS`` record.
    """
    root = project_root or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, *METHOD_A_ARTIFACT.split("/"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Method A artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    records = [r for r in artifact["records"] if r["p"] == REPS]
    if not records:
        raise ValueError(f"no p={REPS} record in {path}")

    def esm(r):
        # exact metrics are nested under the sampling block of the record
        return r["sampling"]["exact_statevector_metrics"]

    def key(r):
        return (-float(esm(r)["optimal_solution_probability"]),
                float(r["best_energy_C0_exact"]), int(r["p"]))

    # each artifact record is one depth; among all records pick by criterion
    chosen = min(artifact["records"], key=key)
    chosen_esm = esm(chosen)
    if int(chosen["p"]) != REPS:
        raise ValueError("selected record is not the expected p="
                         f"{REPS}: p={chosen['p']}")
    theta = [float(v) for v in chosen["best_theta"]]
    if len(theta) != 2 * REPS:
        raise ValueError("theta length mismatch: "
                         f"{len(theta)} != {2 * REPS}")

    inst = core.generate_instance(N, K)
    cost_op0 = core.qubo_to_pauli(inst.mu, inst.Sigma, inst.q, K, 0.0)[0]
    qc_log = q2.bind_theta(p3.build_method_a_circuit(inst, cost_op0, REPS),
                           theta)
    bench = core.exact_benchmark(N, K, inst.mu, inst.Sigma, inst.q)
    ctx = q2.MetricContext.build(inst, 0.0, bench)

    return {
        "artifact_path": os.path.relpath(path, root),
        "artifact_sha256": sha256_file(path),
        "artifact_generated_utc": artifact.get("generated_utc"),
        "record_index": artifact["records"].index(chosen),
        "p": int(chosen["p"]),
        "theta": theta,
        "theta_order": "[gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]",
        "energy_C0_exact": float(chosen["best_energy_C0_exact"]),
        "P_opt_exact": float(chosen_esm["optimal_solution_probability"]),
        "feasibility_rate_exact": float(chosen_esm["feasibility_rate"]),
        "source_params_note": "best_theta read verbatim from the artifact",
        "selection_criterion": REPRESENTATIVE_CRITERION,
        "initial_state": (f"StatePreparation(uniform_weight_state(n={N}, "
                          f"K={K})), {core.uniform_weight_state(N, K).shape[0]}"
                          " amplitudes, weight-K subspace"),
        "mixer": "periodic XY ring mixer (1/2 sum (XX+YY)), "
                 "objective-only cost Hamiltonian (lam=0)",
        "inst": inst,
        "cost_op0": cost_op0,
        "qc_log": qc_log,
        "benchmark": bench,
        "ctx": ctx,
        "optimal_index": int(core.assets_to_index(bench["optimal_assignment"])),
    }


# ---------------------------------------------------------------------------
# Measurement decoding (project convention)
# ---------------------------------------------------------------------------
def bitstring_to_index(bitstring: str, n: int = N) -> int:
    """Qiskit/Aer count key -> project basis index.

    Aer writes the highest qubit leftmost, and the project uses little-endian
    ``index = sum_i x_i 2^i`` with qubit 0 the least significant bit, so the
    identity ``int(key, 2)`` is the correct mapping. Rejects malformed keys.
    """
    s = str(bitstring).replace(" ", "")
    if len(s) != n or set(s) - {"0", "1"}:
        raise ValueError(f"expected a {n}-bit binary string, got {bitstring!r}")
    return int(s, 2)


def index_to_bitstring(index: int, n: int = N) -> str:
    """Project basis index -> Qiskit/Aer count key (highest qubit leftmost)."""
    if not 0 <= index < (1 << n):
        raise ValueError(f"index {index} out of range for n={n}")
    return format(index, f"0{n}b")


def known_bitstring_check() -> Dict:
    """Independent, hand-derived decoding check.

    The exact optimum has assets [1, 5, 6, 8, 10, 11] in a 12-bit instance.
    Hand computation: 2+32+64+256+1024+2048 = 3426, and the Qiskit read order
    (q11 ... q0) therefore writes x as 1,1,0,1,0,1,1,0,0,0,1,0.
    """
    assets = [1, 5, 6, 8, 10, 11]
    x = [0] * N
    for i in assets:
        x[i] = 1
    index = core.assets_to_index(x)
    key = "".join(str(x[i]) for i in range(N - 1, -1, -1))
    reverse_key = key[::-1]
    return {
        "expected_index": 3426,
        "expected_qiskit_key": key,
        "computed_index": int(index),
        "computed_key": index_to_bitstring(index, N),
        "decode_roundtrip_ok": bool(bitstring_to_index(key, N) == index),
        "key_matches_expected": bool(key == "110101100010" and index == 3426),
        "index_to_assets_ok": bool(
            sorted(np.nonzero(core.index_to_assets(index, N))[0].tolist())
            == assets),
        "reversed_key_gives_different_index":
            bool(int(reverse_key, 2) != index),
        "reversed_key_value": int(reverse_key, 2),
        "note": ("the Downloads Phase 4 scratch scripts used int(key[::-1], 2), "
                 "which decodes to index "
                 f"{int(reverse_key, 2)} instead of {index}"),
    }


def counts_to_weights(counts: Dict[str, int], n: int = N) -> Dict[int, int]:
    """Aer count key -> project index -> count."""
    out: Dict[int, int] = {}
    for key, c in counts.items():
        idx = bitstring_to_index(key, n)
        out[idx] = out.get(idx, 0) + int(c)
    return out


def metrics_from_aer_counts(counts: Dict[str, int],
                            ctx: q2.MetricContext) -> Dict:
    """Phase 3 metric definitions applied to an Aer count dictionary."""
    weights = {float(k): float(v) for k, v in
               counts_to_weights(counts, ctx.n).items()}
    return q2.aggregate_metrics(weights, float(sum(weights.values())), ctx)


# ---------------------------------------------------------------------------
# FakeBrisbane subgraph selection
# ---------------------------------------------------------------------------
def backend_edges(backend) -> List[Tuple[int, int]]:
    cfg = backend.configuration()
    return [tuple(int(v) for v in e) for e in cfg.coupling_map]


def select_subgraph(edges: Sequence[Tuple[int, int]],
                    num_qubits: int, size: int = N) -> Dict:
    """Deterministically pick a connected ``size``-node induced subgraph.

    Greedy edge-maximising growth over every starting seed; ties broken by the
    lexicographically smallest layout. Returns the layout and the induced
    edges. The result for FakeBrisbane is a 12-node path with 11 induced edges
    (the densest 12-node region the heavy-hex map contains).
    """
    adj: Dict[int, set] = {}
    for a, c in edges:
        adj.setdefault(a, set()).add(c)
        adj.setdefault(c, set()).add(a)
    edge_set = {tuple(sorted((int(a), int(c)))) for a, c in edges}

    def n_edges(nodes) -> int:
        return sum(1 for a, c in edge_set if a in nodes and c in nodes)

    seeds = [tuple(sorted((a, c))) for a, c in edges]
    seeds += [(r,) for r in range(num_qubits)]
    best_e, best_layout = -1, None
    for seed in seeds:
        nodes = set(seed)
        while len(nodes) < size:
            cand = {v for u in nodes for v in adj.get(u, ()) if v not in nodes}
            if not cand:
                break
            pick = min(cand, key=lambda v: (-len(adj[v] & nodes), v))
            if len(adj[pick] & nodes) == 0:
                break
            nodes.add(pick)
        if len(nodes) == size:
            e = n_edges(nodes)
            layout = sorted(nodes)
            if e > best_e or (e == best_e and
                              (best_layout is None or layout < best_layout)):
                best_e, best_layout = e, layout
    if best_layout is None:
        raise RuntimeError("could not find a connected subgraph of the "
                           "requested size")
    return {
        "layout": best_layout,
        "induced_edges": sorted([tuple(sorted(e)) for e in
                                 edge_set
                                 if e[0] in set(best_layout)
                                 and e[1] in set(best_layout)]),
        "num_induced_edges": int(best_e),
        "num_logical_qubits": int(size),
    }


def subgraph_coupling_map(edges: Sequence[Tuple[int, int]],
                          layout: Sequence[int]) -> CouplingMap:
    nodes = set(int(v) for v in layout)
    cm = CouplingMap(couplinglist=[tuple(sorted((int(a), int(c))))
                                   for a, c in edges
                                   if int(a) in nodes and int(c) in nodes])
    for q in sorted(nodes):
        if q not in cm.physical_qubits:
            cm.add_physical_qubit(q)
    return cm


# ---------------------------------------------------------------------------
# Resource analysis
# ---------------------------------------------------------------------------
def gate_report(qc: QuantumCircuit) -> Dict:
    ops = {str(k): int(v) for k, v in qc.count_ops().items()}
    two_qubit: Dict[str, int] = {}
    explicit_swap = 0
    for instr in qc.data:
        name = getattr(instr.operation, "name", "")
        nq = int(getattr(instr.operation, "num_qubits", 0))
        if nq == 2:
            two_qubit[name] = two_qubit.get(name, 0) + 1
            if name == "swap":
                explicit_swap += 1
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "num_qubits": int(qc.num_qubits),
        "operations": ops,
        "two_qubit_gate_count": int(sum(two_qubit.values())),
        "two_qubit_gates_by_name": two_qubit,
        "explicit_swap_count": int(explicit_swap),
    }


def nonlocal_pair_report(qc: QuantumCircuit, layout: Sequence[int],
                         edges: Sequence[Tuple[int, int]]) -> Dict:
    """Logical 2-qubit pairs that are not adjacent under ``layout``.

    Every such pair needs at least one SWAP of routing to execute, which is why
    post-transpilation gate counts exceed the logical ones even when no explicit
    ``swap`` instruction survives basis decomposition.
    """
    edge_set = {tuple(sorted((int(a), int(c)))) for a, c in edges}
    layout = [int(v) for v in layout]
    total = 0
    nonlocal_count = 0
    offenders = 0
    for instr in qc.data:
        if int(getattr(instr.operation, "num_qubits", 0)) != 2:
            continue
        total += 1
        q = [layout[qc.find_bit(x).index] for x in instr.qubits]
        if tuple(sorted(q)) not in edge_set:
            nonlocal_count += 1
            offenders += 1
    return {
        "logical_two_qubit_gates": int(total),
        "non_adjacent_under_layout": int(nonlocal_count),
        "adjacent_under_layout": int(total - nonlocal_count),
        "routing_required_fraction": (float(nonlocal_count) / total
                                      if total else 0.0),
        "note": ("a non-adjacent logical pair must be routed; routing shows up "
                 "as additional 2-qubit gates and depth after transpilation, "
                 "not necessarily as explicit swap instructions"),
        "_sample_offenders": int(offenders),
    }


def _validate_transpiled(qc: QuantumCircuit, cmap: CouplingMap,
                         basis_gates: Sequence[str]) -> Dict:
    """Check that every 2q gate sits on a coupling edge and 1q gates are in
    the target basis."""
    edge_set = {tuple(sorted(e)) for e in cmap.get_edges()}
    bad_edges = 0
    for instr in qc.data:
        if int(getattr(instr.operation, "num_qubits", 0)) == 2:
            q = tuple(sorted(qc.find_bit(x).index for x in instr.qubits))
            if q not in edge_set:
                bad_edges += 1
    allowed = set(basis_gates)
    allowed |= {"measure", "reset", "barrier"}
    bad_basis = [str(k) for k in qc.count_ops() if str(k) not in allowed]
    return {
        "all_two_qubit_gates_on_coupling_edges": bool(bad_edges == 0),
        "num_offending_two_qubit_gates": int(bad_edges),
        "all_operations_in_target_basis": bool(not bad_basis),
        "offending_operations": bad_basis,
        "layout_present": qc.layout is not None,
        "final_layout": (qc.layout.final_index_layout(filter_ancillas=True)
                         if qc.layout is not None else None),
    }


def resource_analysis(sel: Dict, backend) -> Dict:
    """Experiment A: logical vs transpiled resources, with state prep split."""
    inst, theta, reps = sel["inst"], sel["theta"], sel["p"]
    qc_log = sel["qc_log"]
    qc_dec = qc_log.decompose(reps=DECOMPOSE_REPS)
    bg = list(backend.basis_gates)
    edges = backend_edges(backend)
    sg = select_subgraph(edges, backend.num_qubits, N)
    layout = sg["layout"]
    sub = subgraph_coupling_map(edges, layout)

    # state preparation in isolation (same transpile settings)
    sp = QuantumCircuit(N)
    sp.append(StatePreparation(core.uniform_weight_state(N, K)), range(N))
    sp_dec = sp.decompose(reps=DECOMPOSE_REPS)

    def _tp(circuit, coupling_map, lay):
        t0 = time.perf_counter()
        out = transpile(circuit, coupling_map=coupling_map, basis_gates=bg,
                        initial_layout=lay, optimization_level=OPT_LEVEL,
                        seed_transpiler=SEED_TRANSPILER)
        return out, time.perf_counter() - t0

    failures: Dict[str, str] = {}
    # 1. subgraph (primary, also used for the noisy run)
    try:
        tq_sub, t_sub = _tp(qc_dec, sub, layout)
        val_sub = _validate_transpiled(tq_sub, sub, bg)
    except Exception as exc:  # noqa: BLE001
        tq_sub, t_sub, val_sub = None, None, None
        failures["subgraph"] = f"{type(exc).__name__}: {exc}"
    # 2. state preparation alone on the subgraph
    try:
        sp_tq, t_sp = _tp(sp_dec, sub, layout)
        sp_logical = gate_report(sp_dec)
    except Exception as exc:  # noqa: BLE001
        sp_tq, t_sp, sp_logical = None, None, None
        failures["state_preparation"] = f"{type(exc).__name__}: {exc}"
    # 3. full device (secondary: quantifies real heavy-hex routing)
    try:
        full_cmap = CouplingMap(couplinglist=edges)
        for q in range(int(backend.num_qubits)):
            if q not in full_cmap.physical_qubits:
                full_cmap.add_physical_qubit(q)
        tq_full, t_full = _tp(qc_dec, full_cmap, None)
        val_full = _validate_transpiled(tq_full, full_cmap, bg)
    except Exception as exc:  # noqa: BLE001
        tq_full, t_full, val_full = None, None, None
        failures["full_device"] = f"{type(exc).__name__}: {exc}"

    logical_opaque = gate_report(qc_log)
    logical_dec = gate_report(qc_dec)
    routing = nonlocal_pair_report(qc_dec, layout, edges)

    def _overheads(rep: Optional[Dict], elapsed: Optional[float]) -> Optional[Dict]:
        if rep is None:
            return None
        return {
            "additional_depth": int(rep["depth"] - logical_dec["depth"]),
            "depth_overhead_relative": (float(rep["depth"]) /
                                        logical_dec["depth"]),
            "additional_two_qubit_gates": int(
                rep["two_qubit_gate_count"] - logical_dec["two_qubit_gate_count"]),
            "two_qubit_overhead_relative": (
                float(rep["two_qubit_gate_count"]) /
                logical_dec["two_qubit_gate_count"]
                if logical_dec["two_qubit_gate_count"] else None),
            "transpiler_runtime_s": elapsed,
        }

    sp_rep = gate_report(sp_tq) if sp_tq is not None else None
    return {
        "backend": {
            "name": str(backend.name),
            "num_qubits": int(backend.num_qubits),
            "basis_gates": bg,
            "num_coupling_edges": len(edges),
            "coupling_map_edges": [list(e) for e in edges],
            "target_operation_names": sorted(backend.target.operation_names),
            "has_instruction_durations": bool(backend.target.durations()
                                              is not None),
        },
        "transpiler": {
            "optimization_level": OPT_LEVEL,
            "seed_transpiler": SEED_TRANSPILER,
            "decompose_reps": DECOMPOSE_REPS,
            "qiskit_convention": ("logical qubit i -> physical qubit "
                                  f"layout[i] = {layout}"),
        },
        "subgraph": {
            "layout": layout,
            "num_induced_edges": sg["num_induced_edges"],
            "induced_edges": [list(e) for e in sg["induced_edges"]],
            "construction": ("greedy edge-maximising growth over every seed; "
                             "densest 12-node connected region of the map"),
        },
        "circuit_identity": (f"Method {METHOD} constraint-preserving XY-QAOA, "
                             f"p={reps}, n={N}, K={K}"),
        "metrics_scope": {
            "logical_opaque": ("as built: StatePreparation and "
                               "PauliEvolutionGate counted as single opaque "
                               "instructions, so 2-qubit counts are NOT "
                               "available here"),
            "logical_decomposed": (f"after QuantumCircuit.decompose(reps="
                                   f"{DECOMPOSE_REPS}); this is the basis for "
                                   "all overhead ratios"),
            "state_preparation": "decomposed and transpiled in isolation",
            "note": ("pre-transpilation metrics are logical_*; "
                     "post-transpilation metrics are *_subgraph and "
                     "*_full_device. They are never mixed."),
        },
        "logical_opaque": logical_opaque,
        "logical_decomposed": logical_dec,
        "state_preparation_logical_decomposed": sp_logical,
        "state_preparation_transpiled": sp_rep,
        "state_preparation_contribution": (
            {
                "share_of_transpiled_depth": (
                    float(sp_rep["depth"]) / tq_sub.depth()
                    if sp_rep and tq_sub is not None and tq_sub.depth() else None),
                "share_of_transpiled_size": (
                    float(sp_rep["size"]) / tq_sub.size()
                    if sp_rep and tq_sub is not None and tq_sub.size() else None),
                "share_of_transpiled_two_qubit_gates": (
                    float(sp_rep["two_qubit_gate_count"]) /
                    gate_report(tq_sub)["two_qubit_gate_count"]
                    if sp_rep and tq_sub is not None else None),
            } if sp_rep is not None and tq_sub is not None else None),
        "routing_analysis": routing,
        "transpiled_subgraph": (gate_report(tq_sub) if tq_sub is not None
                                else None),
        "transpiled_subgraph_validation": val_sub,
        "transpiled_full_device": (gate_report(tq_full) if tq_full is not None
                                   else None),
        "transpiled_full_device_validation": val_full,
        "overheads_subgraph": _overheads(gate_report(tq_sub)
                                         if tq_sub is not None else None, t_sub),
        "overheads_full_device": _overheads(gate_report(tq_full)
                                            if tq_full is not None else None,
                                            t_full),
        "state_preparation_transpiler_runtime_s": t_sp,
        "transpiler_runtimes_s": {"subgraph": t_sub, "full_device": t_full,
                                  "state_preparation": t_sp},
        "failures": failures,
        "interpretation": (
            "No explicit swap instructions survive basis decomposition, so the "
            "transpiled 2-qubit gate count and depth carry the routing cost. "
            "The custom fixed-weight initial state is the dominant contributor "
            "(see state_preparation_contribution)."),
    }


# ---------------------------------------------------------------------------
# Experiment B: ideal and noisy sampling
# ---------------------------------------------------------------------------
def ideal_reference(sel: Dict, shots: int = SHOTS_IDEAL,
                    seed: int = SEED_IDEAL) -> Dict:
    """Noiseless reference for the *actual* Method A gate-based circuit.

    The primary ideal reference is the exact Statevector of the decomposed
    logical circuit that is transpiled and later run under noise, so ideal vs
    noisy is apples-to-apples. The Phase 3 weighted-subspace distribution is
    computed separately and reported as a ``trotterization_gap`` (the Method A
    Qiskit circuit uses a per-edge Trotterized XY mixer, whereas Phase 3's
    metrics came from the exact matrix exponential of the XY-ring sum); this is
    an implementation gap, not hardware noise, and never a validation failure.
    """
    qc_dec = sel["qc_log"].decompose(reps=DECOMPOSE_REPS)
    diag = core.operator_diagonal(sel["cost_op0"])
    t0 = time.perf_counter()
    exact = Statevector(qc_dec)
    t_exact = time.perf_counter() - t0
    probs = np.asarray(exact.probabilities(), dtype=float)

    # Phase 3 reference: exact weighted-subspace simulator (exact matrix exp)
    subspace = p3.WeightSubspaceSimulator(N, K, diag, REPS)
    sub_amp = subspace.statevector(sel["theta"])
    feasible = subspace.feasible
    sub_probs = np.abs(sub_amp) ** 2
    sub_probs = sub_probs / sub_probs.sum()
    exact_on_feasible = probs[feasible]
    infeasible_mass = float(probs.sum() - exact_on_feasible.sum())
    sp = np.zeros(probs.shape[0], dtype=float)
    sp[feasible] = sub_probs

    rng = np.random.default_rng(seed)
    draws = rng.choice(probs.shape[0], size=shots, p=probs / probs.sum())
    values, counts = np.unique(draws, return_counts=True)
    index_counts = {int(v): int(c) for v, c in zip(values, counts)}

    weights = {float(i): float(c) for i, c in index_counts.items()}
    metrics = q2.aggregate_metrics(weights, float(sum(weights.values())),
                                   sel["ctx"])

    qc_meas = qc_dec.copy()
    qc_meas.measure_all()
    aer = AerSimulator(method="statevector")
    t1 = time.perf_counter()
    aer_counts = aer.run(qc_meas, shots=shots, seed_simulator=seed).result()
    aer_dict = aer_counts.get_counts()
    t_aer = time.perf_counter() - t1
    aer_weights = counts_to_weights(aer_dict, N)

    return {
        "method": ("exact Statevector probabilities of the decomposed Method A "
                   "gate-based circuit (no measurement), plus a seeded "
                   "multinomial draw and an independent Aer statevector run"),
        "ideal_reference_definition": ("noiseless statevector of the actual "
                                       "Method A circuit (Trotterized per-edge "
                                       "XY mixer) that is transpiled and run "
                                       "under noise"),
        "exact_vs_sampled": ("primary ideal metrics are from the exact circuit "
                             "distribution; sampled metrics use the same shot "
                             "count as the noisy run"),
        "shots": int(shots),
        "seed_simulator": int(seed),
        "exact_runtime_s": t_exact,
        "aer_runtime_s": t_aer,
        "exact_probabilities": [float(v) for v in probs],
        "subspace_exact_probabilities": [float(v) for v in sp],
        "exact_statevector_energy_C0": float(np.sum(probs * diag)),
        "metrics_exact": q2.metrics_from_probs(probs, sel["ctx"]),
        "metrics_subspace_exact": q2.metrics_from_probs(sp, sel["ctx"]),
        "metrics_sampled": metrics,
        "aer_counts_total": int(sum(aer_dict.values())),
        "aer_sampled_metrics": q2.aggregate_metrics(
            {float(k): float(v) for k, v in aer_weights.items()},
            float(sum(aer_weights.values())), sel["ctx"]),
        "aer_raw_counts": {k: int(v) for k, v in sorted(aer_dict.items())},
        "validation": {
            "circuit_statevector_norm": float(exact.probabilities().sum()),
            "circuit_infeasible_mass": infeasible_mass,
            "circuit_support_is_weight_K": bool(infeasible_mass < 1e-9),
            "feasibility_rate_is_one": bool(
                abs(q2.metrics_from_probs(probs, sel["ctx"])
                    ["feasibility_rate"] - 1.0) < 1e-9),
        },
        "trotterization_gap": {
            "definition": ("distance between the noiseless Method A circuit "
                           "distribution (per-edge Trotterized XY mixer) and "
                           "the Phase 3 exact weighted-subspace distribution "
                           "(exact matrix exponential of the XY-ring sum)"),
            "is_hardware_noise": False,
            "max_abs_probability_difference": float(
                np.max(np.abs(sub_probs - exact_on_feasible))),
            "total_variation": float(
                0.5 * np.abs(sub_probs - exact_on_feasible).sum()),
            "fidelity_on_feasible_subspace": float(
                np.sum(np.sqrt(np.clip(sub_probs, 0, None)
                               * np.clip(exact_on_feasible, 0, None))) ** 2),
            "optimal_probability_circuit": float(probs[sel["optimal_index"]]),
            "optimal_probability_subspace_exact": float(
                sp[sel["optimal_index"]]),
            "energy_C0_circuit_statevector": float(np.sum(probs * diag)),
            "energy_C0_subspace_exact": float(np.sum(sp * diag)),
            "phase3_reported_optimal_probability": float(sel["P_opt_exact"]),
            "note": ("the circuit realization is a first-order Trotter product "
                     "of the XY-ring terms; the Phase 3 optimization used the "
                     "exact exponential. Recorded here for transparency and "
                     "kept separate from the noisy-vs-ideal comparison."),
        },
        "circuit": {"depth": int(qc_dec.depth()), "size": int(qc_dec.size())},
    }


def noisy_child(out_path: str, shots: int = SHOTS_NOISY,
                seed: int = SEED_NOISY) -> int:
    """Child-process entry point for the noisy run (see `run_noisy_isolated`)."""
    t_start = time.perf_counter()
    sel = load_method_a()
    backend = _fake_brisbane()
    edges = backend_edges(backend)
    sg = select_subgraph(edges, backend.num_qubits, N)
    layout = sg["layout"]
    sub = subgraph_coupling_map(edges, layout)
    qc_dec = sel["qc_log"].decompose(reps=DECOMPOSE_REPS)
    t0 = time.perf_counter()
    tq = transpile(qc_dec, coupling_map=sub, basis_gates=list(backend.basis_gates),
                   initial_layout=layout, optimization_level=OPT_LEVEL,
                   seed_transpiler=SEED_TRANSPILER)
    t_tp = time.perf_counter() - t0
    validation = _validate_transpiled(tq, sub, list(backend.basis_gates))
    qc_meas = tq.copy()
    qc_meas.measure_all()
    sim = AerSimulator.from_backend(backend)
    t1 = time.perf_counter()
    result = sim.run(qc_meas, shots=int(shots), seed_simulator=int(seed)).result()
    counts = result.get_counts()
    t_run = time.perf_counter() - t1
    payload = {
        "status": "completed",
        "shots": int(shots),
        "seed_simulator": int(seed),
        "backend": str(backend.name),
        "noise_model_source": "AerSimulator.from_backend(FakeBrisbane())",
        "transpiler_runtime_s": t_tp,
        "simulation_runtime_s": t_run,
        "total_runtime_s": time.perf_counter() - t_start,
        "circuit_depth": int(tq.depth()),
        "circuit_size": int(tq.size()),
        "circuit_ops": {str(k): int(v) for k, v in tq.count_ops().items()},
        "transpiled_validation": validation,
        "raw_counts": {k: int(v) for k, v in sorted(counts.items())},
        "counts_total": int(sum(counts.values())),
        "python": platform.python_version(),
        "qiskit_sim_method": str(sim.options.method)
        if hasattr(sim.options, "method") else "automatic",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _fake_brisbane():
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
    return FakeBrisbane()


def noisy_experiment(shots: int = SHOTS_NOISY, seed: int = SEED_NOISY,
                     timeout_s: float = NOISY_TIMEOUT_S,
                     workdir: Optional[str] = None) -> Dict:
    """Run the noisy sampling in a child process under a genuine timeout."""
    import subprocess
    import sys

    root = workdir or os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root, ".phase4_noisy_child.json")
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, os.path.basename(__file__))
    cmd = [sys.executable, "-u", script, "noisy-child", out_path,
           str(int(shots)), str(int(seed))]
    shown = (f"{os.path.basename(sys.executable)} -u {os.path.basename(script)} "
             f"noisy-child <out.json> {shots} {seed}")
    # never mistake a stale child output for this run's measurement
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    t0 = time.perf_counter()
    timed_out = False
    err = None
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              timeout=timeout_s)
        exit_code = int(proc.returncode)
        if exit_code != 0:
            err = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        err = ("child process exceeded the "
               f"{timeout_s:g} s wall-clock timeout; partial stdout: "
               f"{(exc.stdout or '')!r}")
    elapsed = time.perf_counter() - t0

    payload = None
    if not timed_out and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            err = f"unreadable child output: {exc}"
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass

    if timed_out or payload is None:
        return {
            "status": "timeout" if timed_out else "error",
            "command": shown,
            "timeout_s": float(timeout_s),
            "elapsed_s": round(elapsed, 3),
            "exit_code": exit_code,
            "error": (err or "")[:2000],
            "raw": None,
            "completed": False,
        }
    return {
        "status": "completed",
        "command": shown,
        "timeout_s": float(timeout_s),
        "elapsed_s": round(elapsed, 3),
        "exit_code": exit_code,
        "error": None,
        "raw": payload,
        "completed": bool(payload.get("counts_total", 0) > 0),
    }


def compare_ideal_noisy(sel: Dict, ideal: Dict, noisy: Dict) -> Dict:
    """Ideal vs noisy comparison under clearly stated comparison semantics."""
    if not noisy.get("completed"):
        return {
            "status": "unavailable",
            "reason": (f"noisy run status={noisy.get('status')} "
                       f"elapsed={noisy.get('elapsed_s')}s "
                       f"timeout={noisy.get('timeout_s')}s"),
            "comparison_semantics": None,
        }
    ctx = sel["ctx"]
    exact_probs = np.asarray(ideal["exact_probabilities"], dtype=float)
    noisy_weights = counts_to_weights(noisy["raw"]["raw_counts"], N)
    noisy_shots = float(sum(noisy_weights.values()))
    noisy_probs = np.zeros_like(exact_probs)
    for i, w in noisy_weights.items():
        noisy_probs[i] = w / noisy_shots

    tv_noisy = float(0.5 * np.abs(exact_probs - noisy_probs).sum())

    # equal-shot sampled comparison: draw the ideal at the noisy shot count
    # with a fixed seed, so sampled-vs-sampled uses identical shots
    ideal_equal = q2.sample_counts_exact(exact_probs, int(noisy_shots),
                                         seed=int(SEED_IDEAL))
    ip = np.zeros_like(exact_probs)
    for i, w in ideal_equal.items():
        ip[int(i)] = float(w) / noisy_shots
    tv_sampled = float(0.5 * np.abs(ip - noisy_probs).sum())
    tv_sampled_semantics = (
        "sampled (ideal exact distribution, "
        f"{int(noisy_shots)} shots, seed {int(SEED_IDEAL)}) vs sampled "
        f"(noisy, {int(noisy_shots)} shots, seed {int(SEED_NOISY)})")
    metrics_ideal_equal = q2.aggregate_metrics(
        {float(k): float(v) for k, v in ideal_equal.items()},
        noisy_shots, ctx)

    metrics_noisy = q2.aggregate_metrics(
        {float(k): float(v) for k, v in noisy_weights.items()},
        noisy_shots, ctx)
    return {
        "status": "completed",
        "comparison_semantics": {
            "primary": ("exact-distribution (ideal circuit statevector) vs "
                        "sampled-distribution (noisy shots)"),
            "secondary": tv_sampled_semantics,
            "note": ("TV against an exact distribution carries a positive "
                     "sampling-noise bias; the equal-shot sampled-vs-sampled "
                     "value is the fairer measure"),
        },
        "shots_ideal": int(ideal["shots"]),
        "shots_noisy": int(noisy["raw"]["shots"]),
        "seed_ideal": int(SEED_IDEAL),
        "seed_noisy": int(SEED_NOISY),
        "metrics_ideal_exact": ideal["metrics_exact"],
        "metrics_ideal_sampled": ideal["metrics_sampled"],
        "metrics_ideal_equal_sampled": metrics_ideal_equal,
        "metrics_noisy_sampled": metrics_noisy,
        "total_variation_exact_vs_noisy": tv_noisy,
        "total_variation_sampled_vs_sampled": tv_sampled,
        "feasibility_rate_drop": float(
            ideal["metrics_exact"]["feasibility_rate"] -
            metrics_noisy["feasibility_rate"]),
        "optimal_probability_ideal_exact": float(
            ideal["metrics_exact"]["optimal_solution_probability"]),
        "optimal_probability_noisy": float(
            metrics_noisy["optimal_solution_probability"]),
        "ideal_is_constrained_by_construction": True,
        "noisy_is_not": True,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(res: Dict, out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)

    # 1. logical vs transpiled resources
    ra = res["circuit_resources"]
    l_dec = ra["logical_decomposed"]
    sub = ra["transpiled_subgraph"]
    full = ra["transpiled_full_device"]
    sp = ra["state_preparation_transpiled"]
    if sub is not None:
        labels = ["logical\n(decomposed)", "state prep\n(transpiled)",
                  "transpiled\n(12q subgraph)", "transpiled\n(full device)"]
        depth = [l_dec["depth"], sp["depth"] if sp else 0, sub["depth"],
                 full["depth"] if full else 0]
        two_q = [l_dec["two_qubit_gate_count"],
                 sp["two_qubit_gate_count"] if sp else 0,
                 sub["two_qubit_gate_count"],
                 full["two_qubit_gate_count"] if full else 0]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].bar(x, depth, color=["#8ecae6", "#ffb703", "#219ebc", "#023047"])
        ax[0].set_yscale("log")
        ax[0].set_ylabel("depth (log scale)")
        ax[0].set_title("Circuit depth")
        ax[1].bar(x, two_q, color=["#8ecae6", "#ffb703", "#219ebc", "#023047"])
        ax[1].set_yscale("log")
        ax[1].set_ylabel("two-qubit gates (log scale)")
        ax[1].set_title("Two-qubit gate count")
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels(labels, fontsize=8)
            a.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Method A p={res['p']} logical vs transpiled resources "
                     f"({res['backend']['name']})", fontsize=12)
        fig.tight_layout()
        p = os.path.join(figs, "logical_vs_transpiled_resources.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    # 2. ideal vs noisy metrics
    cmp_ = res.get("comparison")
    if cmp_ and cmp_.get("status") == "completed":
        i_e = cmp_["metrics_ideal_exact"]
        i_s = (cmp_.get("metrics_ideal_equal_sampled")
               or cmp_["metrics_ideal_sampled"])
        n_s = cmp_["metrics_noisy_sampled"]
        series = [
            ("feasibility rate",
             [i_e["feasibility_rate"], i_s["feasibility_rate"],
              n_s["feasibility_rate"]]),
            ("P(optimal)", [i_e["optimal_solution_probability"],
                            i_s["optimal_solution_probability"],
                            n_s["optimal_solution_probability"]]),
            ("mean C0 | feasible\n(normalized score)",
             [i_e["feasible_normalized_score"],
              i_s["feasible_normalized_score"],
              n_s["feasible_normalized_score"]]),
        ]
        x = np.arange(len(series))
        w = 0.26
        fig, ax = plt.subplots(figsize=(12, 4.6))
        for k, (lbl, vals, c) in enumerate(zip(
                ["ideal exact", "ideal sampled\n(equal shots)",
                 "noisy sampled"],
                zip(*[v for _, v in series]),
                ["#2a9d8f", "#8ecae6", "#e76f51"])):
            ax.bar(x + (k - 1) * w, vals, width=w, label=lbl, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in series], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("value")
        ax.legend(fontsize=9)
        ax.set_title(f"Ideal vs FakeBrisbane-noisy sampling - Method A "
                     f"p={res['p']}, {cmp_['shots_noisy']} noisy shots",
                     fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(figs, "ideal_vs_noisy_metrics.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# Child entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 helper entry point")
    sub = parser.add_subparsers(dest="cmd")
    ch = sub.add_parser("noisy-child")
    ch.add_argument("out_path")
    ch.add_argument("shots", type=int)
    ch.add_argument("seed", type=int)
    args = parser.parse_args(argv)
    if args.cmd == "noisy-child":
        return noisy_child(args.out_path, args.shots, args.seed)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
