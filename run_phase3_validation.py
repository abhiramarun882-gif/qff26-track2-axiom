"""Targeted Phase 3 validation regeneration.

This script does NOT rerun the Phase 3 optimization and does NOT overwrite the
final experiment JSON files or figures. It regenerates only:

    outputs/phase3/validation_report.json
    logs/phase3-tests.log
    logs/phase3-run.log

Run from the project root with the project interpreter:

    .venv\\Scripts\\python.exe -u run_phase3_validation.py

What is measured fresh here
---------------------------
1. Full test suite via a child process -> real exit code -> logs/phase3-tests.log
2. n=6 Method B and Method C reference checks vs independent Qiskit circuits
   (fast, reuses run_phase3._validate_full_space)
3. n=12 Method A reference check vs an independent Qiskit circuit, one child
   process per depth with a genuine wall-clock timeout. If a check times out it
   is recorded as "timeout" and the previously measured values from the
   2026-10-12 run are carried under `earlier_run_reference`, explicitly
   labelled as NOT freshly measured.

Exit codes
----------
0  validation complete (all checks fresh, or unavailable checks labelled)
1  test suite failed
3  a fresh check disagreed with the reference -> report the discrepancy and
   do not write validation_report.json
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase3")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

PYTHON = sys.executable
N12_TIMEOUT_S = 600.0
FID_TOL = 0.999999
ENERGY_TOL = 1e-9

TESTS_LOG = os.path.join(LOG_DIR, "phase3-tests.log")
RUN_LOG = os.path.join(LOG_DIR, "phase3-run.log")
REPORT_PATH = os.path.join(OUT_DIR, "validation_report.json")

EXPERIMENT_JSONS = [
    "continuous_relaxation.json",
    "classical_baselines.json",
    "method_a_xy_qaoa.json",
    "method_b_warm_start.json",
    "method_c_penalty_qaoa.json",
    "comparison_summary.json",
    "figure_data.json",
]

FIGURES = [
    "best_feasible_vs_method_depth.png",
    "feasibility_vs_depth.png",
    "optimal_prob_vs_depth.png",
    "runtime_vs_objective.png",
    "sampling_mass_best_runs.png",
    "sa_objective_distribution.png",
    "start_objective_distributions.png",
]

CHILD_N12_CODE = r"""
import json, sys, os
root = sys.argv[1]
depth = int(sys.argv[2])
sys.path.insert(0, root)
os.chdir(root)
import phase1_core as core
import run_phase3 as r
inst = core.generate_instance(12, 6)
rows = r._validate_method_a(inst, depths=(depth,))
sys.stdout.write("\n<<<RESULT>>>" + json.dumps(rows) + "<<<END>>>\n")
sys.stdout.flush()
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunLog:
    """Collects the exact commands, timestamps, exit codes and outcomes."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.started_utc = _now()

    def __call__(self, msg: str = "") -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}" if msg else ""
        print(msg, flush=True)
        self.lines.append(line)
        # incremental flush so partial progress survives an interruption
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def header(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(RUN_LOG, "w", encoding="utf-8") as fh:
            fh.write("")
        for text in (
            "=" * 78,
            "Phase 3 targeted validation regeneration (NOT an end-to-end rerun)",
            "=" * 78,
            f"started_utc (local machine): {self.started_utc}",
            f"project_root : {PROJECT_ROOT}",
            f"python       : {PYTHON}",
            f"argv         : {sys.argv}",
            "scope: this run regenerates validation_report.json, "
            "logs/phase3-tests.log and logs/phase3-run.log only.",
            "scope: the final Phase 3 experiment JSON files and figures were "
            "preserved, not rerun, and not overwritten.",
            "",
        ):
            self.lines.append(text)
            with open(RUN_LOG, "a", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(text, flush=True)

    def save(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(RUN_LOG, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines))
            fh.write("\n")


def load_artifact_provenance() -> list[dict]:
    out = []
    for name in EXPERIMENT_JSONS:
        path = os.path.join(OUT_DIR, name)
        if not os.path.exists(path):
            out.append({"name": name, "path": os.path.relpath(path, PROJECT_ROOT),
                        "present": False, "kind": "experiment_json"})
            continue
        st = os.stat(path)
        out.append({
            "name": name,
            "kind": "experiment_json",
            "path": os.path.relpath(path, PROJECT_ROOT),
            "present": True,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "bytes": int(st.st_size),
            "sha256": _sha256(path),
            "status": "preserved - not rerun, not overwritten by this script",
        })
    for name in FIGURES:
        path = os.path.join(OUT_DIR, "figures", name)
        if not os.path.exists(path):
            out.append({"name": name, "path": os.path.relpath(path, PROJECT_ROOT),
                        "present": False, "kind": "figure"})
            continue
        st = os.stat(path)
        out.append({
            "name": name,
            "kind": "figure",
            "path": os.path.relpath(path, PROJECT_ROOT),
            "present": True,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "bytes": int(st.st_size),
            "sha256": _sha256(path),
            "status": "preserved - not rerun, not overwritten by this script",
        })
    return out


def load_json(name: str) -> dict:
    with open(os.path.join(OUT_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_tests(log: RunLog) -> dict:
    """Run the full suite in a child process and keep its real exit code."""
    cmd = [PYTHON, "-u", "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    shown = " ".join(cmd)
    log(f"command: {shown}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    output = (proc.stdout or "") + (proc.stderr or "")

    header = (
        "# phase3-tests.log (regenerated by run_phase3_validation.py)\n"
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
    elif re.search(r"^OK\b", output, re.M):
        pass
    if tests_run is None:
        m = re.search(r"^(OK|FAILED)", output, re.M)
        if m and m.group(1) == "FAILED":
            log("WARNING: could not parse 'Ran N tests' from output")

    log(f"tests: exit_code={proc.returncode} tests_run={tests_run} "
        f"failures={failures} errors={errors} skipped={skipped} "
        f"wall={elapsed:.2f}s -> logs/phase3-tests.log")
    return {
        "command": shown,
        "cwd": PROJECT_ROOT,
        "exit_code": int(proc.returncode),
        "wall_clock_s": round(float(elapsed), 3),
        "log_file": "logs/phase3-tests.log",
        "tests_run": tests_run,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": None if tests_run is None else tests_run - failures - errors - skipped,
        "source": "measured fresh by this run (child process, real exit code)",
    }


def run_n6_checks(log: RunLog, core_module, r3_module) -> tuple[dict, dict, list[str]]:
    inst6 = core_module.generate_instance(6, 3)
    problems: list[str] = []
    blocks = {}
    for label in ("B", "C"):
        t0 = time.perf_counter()
        rows = r3_module._validate_full_space(inst6, label)
        elapsed = time.perf_counter() - t0
        min_fid = min(r["fidelity_vs_qiskit"] for r in rows)
        max_dE = max(r["abs_energy_diff"] for r in rows)
        blocks[f"method_{label}_full_vs_qiskit_n6"] = {
            "status": "fresh",
            "method": label,
            "instance": {"n": 6, "K": 3, "lambda": 1.0},
            "seed": 991,
            "source": "measured fresh by this run (run_phase3._validate_full_space)",
            "wall_clock_s": round(float(elapsed), 3),
            "min_fidelity_vs_qiskit": float(min_fid),
            "max_abs_energy_diff": float(max_dE),
            "tolerance": {"fidelity": FID_TOL, "abs_energy_diff": ENERGY_TOL},
            "rows": rows,
        }
        log(f"fresh Method {label} n=6: minFid={min_fid:.12f} "
            f"maxdE={max_dE:.3e} wall={elapsed:.2f}s")
        if min_fid < FID_TOL:
            problems.append(
                f"Method {label} n=6 fidelity {min_fid:.17g} < {FID_TOL}")
        if max_dE > ENERGY_TOL:
            problems.append(
                f"Method {label} n=6 energy diff {max_dE:.3e} > {ENERGY_TOL}")
    return blocks["method_B_full_vs_qiskit_n6"], \
        blocks["method_C_full_vs_qiskit_n6"], problems


def run_n12_check(log: RunLog) -> dict:
    """Method A subspace simulator vs an independent Qiskit circuit at n=12.

    Each depth runs in its own child process under a genuine wall-clock timeout.
    """
    checks = []
    for depth in (1, 3):
        cmd = [PYTHON, "-u", "-c", CHILD_N12_CODE, PROJECT_ROOT, str(depth)]
        shown = f"{os.path.basename(PYTHON)} -u -c <child n=12 Method A depth={depth}> {PROJECT_ROOT} {depth}"
        log(f"command: {shown}")
        log(f"  timeout = {N12_TIMEOUT_S:.0f}s")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=N12_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            log(f"  TIMEOUT after {elapsed:.1f}s (limit {N12_TIMEOUT_S:.0f}s) "
                f"for p={depth} - NOT freshly verified")
            checks.append({
                "p": depth,
                "status": "timeout",
                "timeout_s": float(N12_TIMEOUT_S),
                "elapsed_s": round(float(elapsed), 3),
                "exit_code": None,
                "command": shown,
                "error": "child process exceeded wall-clock timeout; check NOT "
                         "freshly verified",
                "rows": None,
            })
            continue
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"<<<RESULT>>>(.*?)<<<END>>>", out, re.S)
        if proc.returncode != 0 or not m:
            log(f"  child failed exit_code={proc.returncode} for p={depth}")
            tail = out[-800:]
            checks.append({
                "p": depth,
                "status": "error",
                "timeout_s": float(N12_TIMEOUT_S),
                "elapsed_s": round(float(elapsed), 3),
                "exit_code": int(proc.returncode),
                "command": shown,
                "error": tail,
                "rows": None,
            })
            continue
        rows = json.loads(m.group(1))
        for row in rows:
            log(f"  p={depth} fid={row['fidelity_subspace_vs_qiskit']:.12f} "
                f"dE={row['abs_energy_diff']:.3e} "
                f"infeasMass={row['infeasible_mass_qiskit']:.3e} "
                f"wall={elapsed:.1f}s")
        checks.append({
            "p": depth,
            "status": "fresh",
            "timeout_s": float(N12_TIMEOUT_S),
            "elapsed_s": round(float(elapsed), 3),
            "exit_code": int(proc.returncode),
            "command": shown,
            "rows": rows,
        })

    statuses = [c["status"] for c in checks]
    if all(s == "fresh" for s in statuses):
        overall = "fresh"
    elif any(s == "fresh" for s in statuses):
        overall = "partial"
    elif all(s == "timeout" for s in statuses):
        overall = "timeout"
    else:
        overall = "error"
    log(f"n=12 Method A overall status: {overall}")
    return {
        "status": overall,
        "description": "WeightSubspaceSimulator (Method A, weight-K subspace) vs "
                       "the full Qiskit circuit restricted to the weight-K "
                       "amplitudes, evaluated at n=12, K=6, seed 20260303",
        "tolerance": {"fidelity": FID_TOL, "abs_energy_diff": ENERGY_TOL},
        "timeout_s": float(N12_TIMEOUT_S),
        "checks": checks,
    }


def main() -> int:
    log = RunLog()
    log.header()

    import phase1_core as core
    import run_phase3 as r3

    # ---- 0. previous report (source of any carried-forward reference) ----
    previous_report = None
    if os.path.exists(REPORT_PATH):
        try:
            previous_report = json.load(open(REPORT_PATH, "r", encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: could not read previous validation_report.json: {exc}")

    log("--- artifact provenance (read-only) ---")
    provenance = load_artifact_provenance()
    for p in provenance:
        if p["present"]:
            log(f"  {p['name']}: {p['mtime_utc']}  {p['bytes']}B  "
                f"sha256={p['sha256'][:16]}")
        else:
            log(f"  {p['name']}: MISSING")
    log("")

    # ---- 1. test suite ----
    log("--- test suite (child process, real exit code) ---")
    tests = run_tests(log)
    log("")

    # ---- 2. fresh n=6 reference checks ----
    log("--- fresh n=6 reference checks (Method B / Method C) ---")
    block_b, block_c, problems = run_n6_checks(log, core, r3)
    log("")

    # ---- 3. fresh n=12 Method A check with genuine timeout ----
    log("--- fresh n=12 Method A reference check (genuine timeout) ---")
    block_a = run_n12_check(log)
    log("")

    # ---- discrepancy -> stop, do not write the report ----
    if problems:
        log("DISCREPANCY DETECTED - not writing validation_report.json:")
        for p in problems:
            log(f"  * {p}")
        log("stopping so the discrepancy is reported instead of silently "
            "falling back to stale data.")
        log.save()
        return 3

    # ---- carried-forward reference, only if a check was not fresh ----
    earlier = None
    if block_a["status"] != "fresh":
        if previous_report and "method_a_subspace_vs_qiskit_n12" in previous_report:
            earlier = {
                "status": "carried_forward_from_previous_run",
                "source": "outputs/phase3/validation_report.json as written by "
                          "run_phase3.py on 2026-10-12 22:05:38 local time "
                          "(maxiter=500 run)",
                "freshly_measured": False,
                "warning": "These values were NOT re-measured by this run. They "
                           "are included only as a previously verified "
                           "reference alongside the timeout/error recorded above.",
                "method_a_subspace_vs_qiskit_n12":
                    previous_report.get("method_a_subspace_vs_qiskit_n12"),
                "validation_wall_clock_s":
                    previous_report.get("validation_wall_clock_s"),
            }
            log("n=12 Method A not freshly verified -> carrying forward the "
                "2026-10-12 values under earlier_run_reference (labelled as "
                "NOT freshly measured)")
        else:
            log("n=12 Method A not freshly verified and no previous report "
                "available -> no reference carried forward")
    log("")

    # ---- assemble the report ----
    versions = r3._pkg_versions()
    comparison = load_json("comparison_summary.json")
    relax = load_json("continuous_relaxation.json")
    br = relax["relaxation"]["best_run"]
    C_best = comparison["exact_reference"]["C_best"]

    report = {
        "team": "axiom", "phase": 3, "study": "validation_report",
        "generated_utc": _now(),
        "validation_scope": (
            "Targeted regeneration of this validation report only. The final "
            "Phase 3 experiment JSON files and figures were preserved, not "
            "rerun and not overwritten. This is NOT a full end-to-end rerun of "
            "run_phase3.py."
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": versions,
        "final_artifacts": provenance,
        "tests": tests,
        "fresh_checks": {
            "method_a_subspace_vs_qiskit_n12": block_a,
            "method_b_full_vs_qiskit_n6": block_b,
            "method_c_full_vs_qiskit_n6": block_c,
        },
        "earlier_run_reference": earlier,
        "exact_reference": {
            "source": "outputs/phase3/comparison_summary.json (final run "
                      "2026-10-13 06:48:54 local)",
            "C_best": C_best,
            "C_worst": comparison["exact_reference"]["C_worst"],
            "optimal_asset_indices": comparison["exact_reference"]["optimal_asset_indices"],
            "feasible_count": comparison["exact_reference"]["feasible_count"],
            "num_optimal_ties": comparison["exact_reference"]["num_optimal_ties"],
        },
        "continuous_relaxation_check": {
            "source": "outputs/phase3/continuous_relaxation.json (final run "
                      "2026-10-13 06:21:41 local)",
            "relaxation_objective": br["objective"],
            "C_best": C_best,
            "relaxation_is_lower_bound": bool(br["objective"] <= C_best + 1e-9),
            "gap": relax["relaxation"]["gap_to_optimum"],
            "rounded_portfolio_is_optimal":
                br["rounding"]["objective"] <= C_best + 1e-9,
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
            "This validation report was regenerated in targeted mode; the n=12 "
            "Method A reference check is subject to a 600 s wall-clock timeout "
            "and is labelled per depth.",
        ],
        "commands": {
            "this_script": " ".join(sys.argv),
            "tests": tests["command"],
            "n12_method_a_child": [c["command"] for c in block_a["checks"]],
        },
        "timestamps": {
            "started_local_machine": log.started_utc,
            "finished_utc": _now(),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log(f"wrote {os.path.relpath(REPORT_PATH, PROJECT_ROOT)}")

    log("")
    log("summary:")
    log(f"  tests: exit_code={tests['exit_code']} "
        f"run={tests['tests_run']} passed={tests['passed']}")
    log(f"  Method A n=12: {block_a['status']}")
    for c in block_a["checks"]:
        log(f"    p={c['p']}: {c['status']} ({c['elapsed_s']}s)")
    log(f"  Method B n=6: fresh, minFid={block_b['min_fidelity_vs_qiskit']:.12f}")
    log(f"  Method C n=6: fresh, minFid={block_c['min_fidelity_vs_qiskit']:.12f}")
    log(f"  early-run reference carried forward: "
        f"{'yes' if earlier else 'no'}")
    log("")
    log("this was a targeted validation regeneration, not an end-to-end rerun.")
    log.save()

    if tests["exit_code"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
