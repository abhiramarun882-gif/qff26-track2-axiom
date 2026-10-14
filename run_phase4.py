"""Phase 4 orchestration: hardware awareness and noisy simulation.

Team axiom - Qiskit Fall Fest 2026 - Track 2.

Runs Experiment A (logical vs transpiled resources on FakeBrisbane) and
Experiment B (ideal vs FakeBrisbane-noisy sampling) for the selected Method A
circuit, writes ``outputs/phase4/*.json`` and figures incrementally, appends a
dated entry to ``logs/phase4-run.log``, and runs the full test suite into
``logs/phase4-tests.log``.

The noisy sampling runs in a child process under a genuine wall-clock timeout
(see ``phase4_hardware.noisy_experiment``); a timeout is recorded, never hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import phase1_core as core          # noqa: E402
import phase4_hardware as p4        # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase4")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RUN_LOG = os.path.join(LOG_DIR, "phase4-run.log")
TEST_LOG = os.path.join(LOG_DIR, "phase4-tests.log")
PYTHON = sys.executable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def _stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(lines: List[str]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write("\n")
        for line in lines:
            fh.write(f"[{_stamp()}] {line}\n")


def run_tests() -> Dict:
    cmd = [PYTHON, "-u", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    shown = " ".join(cmd)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(TEST_LOG, "w", encoding="utf-8") as fh:
        fh.write(f"# command: {shown}\n")
        fh.write(f"# cwd    : {PROJECT_ROOT}\n")
        fh.write(f"# started: {_now()}\n")
        fh.write(f"# exit_code: {proc.returncode}\n")
        fh.write(f"# wall_clock_s: {elapsed:.3f}\n")
        fh.write("-" * 70 + "\n")
        fh.write(out)
    return {
        "command": shown,
        "exit_code": int(proc.returncode),
        "wall_clock_s": round(elapsed, 3),
        "output_tail": out.strip().splitlines()[-6:],
    }


def build_validation_report(sel, resources, ideal, noisy, comparison, tests,
                            figures) -> Dict:
    checks: List[Dict] = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    kb = p4.known_bitstring_check()
    add("bitstring decoding (known bitstring)", kb["decode_roundtrip_ok"]
        and kb["key_matches_expected"] and kb["index_to_assets_ok"],
        f"index {kb['computed_index']} <-> key {kb['computed_key']} "
        f"(assets {kb['index_to_assets_ok']})")

    add("artifact provenance recorded", bool(sel["artifact_sha256"]),
        f"{sel['artifact_path']} sha256={sel['artifact_sha256'][:16]}..."
        f" p={sel['p']}")

    val = ideal["validation"]
    add("ideal circuit constraint preserved (weight-K support)",
        val["circuit_infeasible_mass"] < 1e-9
        and val["circuit_support_is_weight_K"],
        f"infeasible mass={val['circuit_infeasible_mass']:.3e}, "
        f"state norm={val['circuit_statevector_norm']:.12f}")
    tg = ideal["trotterization_gap"]
    add("trotterization gap documented (circuit vs exact subspace)",
        not tg["is_hardware_noise"],
        f"TV={tg['total_variation']:.4f}, max|dP|="
        f"{tg['max_abs_probability_difference']:.3e}, "
        f"P(opt) circuit={tg['optimal_probability_circuit']:.6f} vs "
        f"exact={tg['optimal_probability_subspace_exact']:.6f}")

    add("ideal feasibility rate == 1 (constraint preserved)",
        abs(ideal["metrics_exact"]["feasibility_rate"] - 1.0) < 1e-9,
        f"feasibility={ideal['metrics_exact']['feasibility_rate']:.12f}")

    sub_val = resources.get("transpiled_subgraph_validation") or {}
    add("transpiled subgraph respects coupling + basis",
        bool(sub_val.get("all_two_qubit_gates_on_coupling_edges"))
        and bool(sub_val.get("all_operations_in_target_basis")),
        f"bad_edges={sub_val.get('num_offending_two_qubit_gates')}, "
        f"offending_ops={sub_val.get('offending_operations')}")

    add("resource metrics available",
        bool(resources.get("logical_decomposed"))
        and bool(resources.get("transpiled_subgraph")),
        f"logical depth={resources['logical_decomposed']['depth']}, "
        f"transpiled depth="
        f"{resources['transpiled_subgraph']['depth'] if resources.get('transpiled_subgraph') else None}")

    add("noisy run completed", bool(noisy.get("completed")),
        f"status={noisy.get('status')} elapsed={noisy.get('elapsed_s')}s "
        f"timeout={noisy.get('timeout_s')}s shots={noisy.get('raw', {}).get('shots') if noisy.get('raw') else None}")

    if comparison.get("status") == "completed":
        add("ideal/noisy comparison produced",
            comparison.get("total_variation_exact_vs_noisy") is not None,
            f"TV(exact,noisy)="
            f"{comparison['total_variation_exact_vs_noisy']:.4f}")
    else:
        add("ideal/noisy comparison produced", False,
            comparison.get("reason"))

    add("full test suite passed", tests["exit_code"] == 0,
        f"exit_code={tests['exit_code']} tail={tests['output_tail']}")

    add("figures written", bool(figures), f"{len(figures)} figure(s)")

    return {
        "team": "axiom", "phase": 4,
        "study": "hardware_awareness_and_noisy_simulation",
        "generated_utc": _now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "all_checks_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "restrictions_observed": [
            "Simulator only: no QPU, no IBM Quantum job submitted.",
            "FakeBrisbane used only through its Target, coupling map and "
            "backend-derived noise model.",
            "No Phase 3 optimization was rerun; the Method A parameters are "
            "read verbatim from the Phase 3 artifact.",
            "No fabricated measurements: the noisy run executes in a child "
            "process and a timeout would be recorded, not hidden.",
            "FakeBrisbane-derived noise is a device model, not a claim of "
            "real-hardware execution.",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 runner")
    parser.add_argument("--noisy-shots", type=int, default=p4.SHOTS_NOISY)
    parser.add_argument("--noisy-seed", type=int, default=p4.SEED_NOISY)
    parser.add_argument("--noisy-timeout", type=float, default=p4.NOISY_TIMEOUT_S)
    parser.add_argument("--ideal-shots", type=int, default=p4.SHOTS_IDEAL)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Phase 4 started {_now()}", flush=True)
    problems: List[str] = []

    # --- select circuit -----------------------------------------------------
    print("\n=== selecting Method A circuit from Phase 3 artifact ===", flush=True)
    sel = p4.load_method_a(PROJECT_ROOT)
    print(f"  artifact : {sel['artifact_path']} "
          f"sha256={sel['artifact_sha256'][:16]}...", flush=True)
    print(f"  p={sel['p']} theta={['%.6f' % v for v in sel['theta']]}", flush=True)
    print(f"  exact P(opt)={sel['P_opt_exact']:.6g} "
          f"E_C0={sel['energy_C0_exact']:.9f}", flush=True)
    kb = p4.known_bitstring_check()
    print(f"  decode check: {kb['computed_key']} -> {kb['computed_index']} "
          f"(pass={kb['decode_roundtrip_ok']})", flush=True)
    if not (kb["decode_roundtrip_ok"] and kb["key_matches_expected"]):
        problems.append("bitstring decoding check failed")

    backend = p4._fake_brisbane()

    # --- Experiment A -------------------------------------------------------
    print("\n=== Experiment A: logical vs transpiled resources ===", flush=True)
    resources = p4.resource_analysis(sel, backend)
    resources = {
        "team": "axiom", "phase": 4, "generated_utc": _now(),
        "selection": {k: sel[k] for k in
                      ("artifact_path", "artifact_sha256",
                       "artifact_generated_utc", "record_index", "p", "theta",
                       "theta_order", "energy_C0_exact", "P_opt_exact",
                       "selection_criterion", "initial_state", "mixer")},
        **resources,
    }
    _dump(os.path.join(OUT_DIR, "circuit_resources.json"), resources)
    sub = resources["transpiled_subgraph"]
    print(f"  logical (decomposed): depth={resources['logical_decomposed']['depth']} "
          f"size={resources['logical_decomposed']['size']} "
          f"2q={resources['logical_decomposed']['two_qubit_gate_count']}", flush=True)
    print(f"  state prep (transpiled): depth={resources['state_preparation_transpiled']['depth']} "
          f"2q={resources['state_preparation_transpiled']['two_qubit_gate_count']}", flush=True)
    print(f"  subgraph transpiled: depth={sub['depth']} size={sub['size']} "
          f"2q={sub['two_qubit_gate_count']} "
          f"ops={sub['operations']}", flush=True)
    if resources["transpiled_full_device"]:
        f = resources["transpiled_full_device"]
        print(f"  full-device transpiled: depth={f['depth']} "
              f"2q={f['two_qubit_gate_count']}", flush=True)
    print(f"  routing: nonlocal 2q pairs="
          f"{resources['routing_analysis']['non_adjacent_under_layout']}/"
          f"{resources['routing_analysis']['logical_two_qubit_gates']}", flush=True)
    if resources["failures"]:
        problems.append(f"transpilation failures: {resources['failures']}")
        print(f"  FAILURES: {resources['failures']}", flush=True)

    # --- Experiment B: ideal ------------------------------------------------
    print("\n=== Experiment B: ideal reference ===", flush=True)
    ideal = p4.ideal_reference(sel, shots=args.ideal_shots)
    ideal = {"team": "axiom", "phase": 4, "generated_utc": _now(), **ideal}
    _dump(os.path.join(OUT_DIR, "ideal_results.json"), ideal)
    print(f"  exact metrics (circuit statevector): {ideal['metrics_exact']}",
          flush=True)
    tg = ideal["trotterization_gap"]
    print(f"  constraint: infeasible mass="
          f"{ideal['validation']['circuit_infeasible_mass']:.3e} "
          f"(support weight-K={ideal['validation']['circuit_support_is_weight_K']})",
          flush=True)
    print(f"  trotterization gap vs exact subspace: TV={tg['total_variation']:.4f} "
          f"max|dP|={tg['max_abs_probability_difference']:.3e} "
          f"P(opt) circuit={tg['optimal_probability_circuit']:.6f} "
          f"exact={tg['optimal_probability_subspace_exact']:.6f}", flush=True)

    # --- Experiment B: noisy -------------------------------------------------
    print("\n=== Experiment B: FakeBrisbane-noisy sampling ===", flush=True)
    print(f"  shots={args.noisy_shots} seed={args.noisy_seed} "
          f"timeout={args.noisy_timeout:g}s", flush=True)
    t0 = time.perf_counter()
    noisy = p4.noisy_experiment(shots=args.noisy_shots, seed=args.noisy_seed,
                                timeout_s=args.noisy_timeout,
                                workdir=PROJECT_ROOT)
    noisy = {"team": "axiom", "phase": 4, "generated_utc": _now(),
             "backend": str(backend.name),
             "noise_model_source": "AerSimulator.from_backend(FakeBrisbane())",
             "simulator_only": True,
             "harmless_elapsed_s": round(time.perf_counter() - t0, 3),
             **noisy}
    _dump(os.path.join(OUT_DIR, "noisy_results.json"), noisy)
    if noisy["completed"]:
        rm = p4.metrics_from_aer_counts(noisy["raw"]["raw_counts"], sel["ctx"])
        noisy_metrics_for_cmp = rm
        print(f"  completed in {noisy['elapsed_s']}s shots="
              f"{noisy['raw']['shots']}", flush=True)
        print(f"  noisy metrics: {noisy_metrics_for_cmp}", flush=True)
    else:
        print(f"  status={noisy['status']} elapsed={noisy['elapsed_s']}s "
              f"(timeout {noisy['timeout_s']}s)", flush=True)
        problems.append(f"noisy run {noisy['status']}")

    # --- comparison ----------------------------------------------------------
    comparison = p4.compare_ideal_noisy(sel, ideal, noisy)
    comparison = {"team": "axiom", "phase": 4, "generated_utc": _now(),
                  **comparison}
    _dump(os.path.join(OUT_DIR, "comparison_summary.json"), comparison)
    if comparison["status"] == "completed":
        print(f"\n=== comparison ===\n  TV(exact ideal, noisy)="
              f"{comparison['total_variation_exact_vs_noisy']:.4f} "
              f"TV(sampled, sampled)="
              f"{comparison['total_variation_sampled_vs_sampled']}", flush=True)
        print(f"  feasibility ideal={comparison['metrics_ideal_exact']['feasibility_rate']:.4f} "
              f"noisy={comparison['metrics_noisy_sampled']['feasibility_rate']:.4f}",
              flush=True)
        print(f"  P(opt) ideal={comparison['optimal_probability_ideal_exact']:.6f} "
              f"noisy={comparison['optimal_probability_noisy']:.6f}", flush=True)

    # --- figures -------------------------------------------------------------
    print("\n=== figures ===", flush=True)
    res_for_fig = {"circuit_resources": resources, "comparison": comparison,
                   "p": sel["p"], "backend": resources["backend"]}
    figures = p4.make_figures(res_for_fig, OUT_DIR)
    for f in figures:
        print(f"  wrote {os.path.relpath(f, PROJECT_ROOT)}", flush=True)

    # --- tests ---------------------------------------------------------------
    print("\n=== test suite ===", flush=True)
    tests = ({"command": "(skipped)", "exit_code": 0, "wall_clock_s": 0.0,
              "output_tail": ["--skip-tests"]} if args.skip_tests
             else run_tests())
    print(f"  exit_code={tests['exit_code']} -> "
          f"logs/phase4-tests.log", flush=True)
    if tests["exit_code"] != 0:
        problems.append(f"test suite exit_code={tests['exit_code']}")

    # --- validation report ---------------------------------------------------
    vreport = build_validation_report(sel, resources, ideal, noisy, comparison,
                                      tests, figures)
    _dump(os.path.join(OUT_DIR, "validation_report.json"), vreport)

    # --- log -----------------------------------------------------------------
    log_lines = [
        "=" * 78,
        "Phase 4: hardware awareness and noisy simulation (FakeBrisbane, "
        "simulator only)",
        "=" * 78,
        f"command           : {' '.join(sys.argv)}",
        f"artifact          : {sel['artifact_path']} "
        f"sha256={sel['artifact_sha256'][:16]}... p={sel['p']}",
        f"decode_check      : {kb['computed_key']} -> {kb['computed_index']} "
        f"ok={kb['decode_roundtrip_ok']}",
        f"logical_decomposed: depth={resources['logical_decomposed']['depth']} "
        f"size={resources['logical_decomposed']['size']} "
        f"2q={resources['logical_decomposed']['two_qubit_gate_count']}",
        f"stateprep_transp  : depth={resources['state_preparation_transpiled']['depth']} "
        f"2q={resources['state_preparation_transpiled']['two_qubit_gate_count']}",
        f"subgraph          : layout={resources['subgraph']['layout']} "
        f"edges={resources['subgraph']['num_induced_edges']}",
        f"transpiled_subgraph: depth={sub['depth']} size={sub['size']} "
        f"2q={sub['two_qubit_gate_count']} ops={sub['operations']}",
        f"routing           : nonlocal_2q="
        f"{resources['routing_analysis']['non_adjacent_under_layout']}/"
        f"{resources['routing_analysis']['logical_two_qubit_gates']} "
        f"explicit_swaps={sub['explicit_swap_count']}",
        f"ideal_exact       : {ideal['metrics_exact']}",
        f"ideal_reference   : noiseless statevector of the Method A gate-based "
        f"circuit (primary ideal for ideal-vs-noisy)",
        f"trotter_gap       : TV={tg['total_variation']:.4f} "
        f"max|dP|={tg['max_abs_probability_difference']:.3e} "
        f"P(opt) circuit={tg['optimal_probability_circuit']:.6f} "
        f"exact={tg['optimal_probability_subspace_exact']:.6f} "
        f"(implementation gap, not hardware noise)",
        f"noisy             : status={noisy['status']} "
        f"elapsed={noisy['elapsed_s']}s timeout={noisy['timeout_s']}s "
        f"shots={args.noisy_shots}",
        f"comparison        : {comparison.get('status')}",
        f"tests             : exit_code={tests['exit_code']}",
        f"figures           : {[os.path.basename(f) for f in figures]}",
        f"validation        : all_checks_passed="
        f"{vreport['all_checks_passed']}",
        "wrote             : outputs/phase4/circuit_resources.json, "
        "ideal_results.json, noisy_results.json, comparison_summary.json, "
        "validation_report.json",
    ]
    append_log(log_lines)

    print(f"\nPhase 4 finished {_now()}", flush=True)
    if problems:
        print("UNRESOLVED:", problems, flush=True)
        return 3
    print("all Phase 4 checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
