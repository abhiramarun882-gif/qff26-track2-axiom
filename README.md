# axiom · Qiskit Fall Fest 2026 · Track 2 (Quantum Optimization)

| | |
|---|---|
| **Team** | axiom |
| **Members** | Abhiram Arun N S V (`abhiramarun882@gmail.com`), Sree Sruthi Alur |
| **Track** | 2 · Quantum Optimization |
| **Level reached** | Intermediate |
| **Code files** | `portfolio_qaoa.ipynb`; `phase1_core.py` … `phase4_hardware.py`; `run_phase1.py` … `run_phase4.py` |
| **Slides** | `slides.pdf` (8 slides) |

## Summary

We solve the Track 2 cardinality-constrained portfolio problem for the
Intermediate instance `n=12, K=6` (exactly 6 of 12 assets) with three quantum
variational methods, and compare them against classical baselines on simulators.
**No quantum advantage was demonstrated:** classical greedy selection,
feasible-swap simulated annealing and relaxation + rounding all reach the exact
optimum `-1.0663920923466492`, and the quantum methods match that solution
quality but do not beat it. We then transpile the best constraint-preserving
circuit for the IBM `FakeBrisbane` device model and show that, at the resulting
depth, hardware-like noise erases the signal.

## Approach

- **Problem / encoding.** `C(x) = q·xᵀΣx − μᵀx + λ(Σxᵢ − K)²`; asset `i` ↔ qubit
  `i`, state index `s = Σ xᵢ 2^i` (qubit 0 = LSB), and `xᵢ = (I − Zᵢ)/2`.
- **Beginner (mandatory).** Standard transverse-field QAOA on `n=6, K=3`
  (`H_M = Σ Xᵢ`, initial `|+⟩^n`), plus an exact penalty-threshold sweep.
- **Intermediate — three routes.** **(A)** constraint-preserving XY-QAOA
  (XY-ring mixer + uniform weight-K state, no penalty, every sample feasible);
  **(B)** warm-start QAOA initialised from the continuous relaxation;
  **(C)** standard penalised QAOA at `λ = 1`. All optimised with COBYLA
  multistart (`rhobeg=1.0`, `maxiter=1500`).
- **Product formula.** Method A's ring XY mixer is non-commuting, so the
  hardware-realizable circuit is a **first-order Trotterization** of the exact
  matrix exponential used during optimisation — a gap we quantify separately
  from noise.
- **Error mitigation.** None (no QPU run). The noisy study uses
  `AerSimulator.from_backend(FakeBrisbane())`, i.e. a device-model noise model.

## Results

| Quantity | Result | Reference | Difference |
|---|---|---|---|
| Intermediate exact optimum `C0` | `-1.0663920923466492` | exhaustive over 924 portfolios | 0 (by construction) |
| Greedy / SA feasible-swap | `-1.066392` | exact optimum | 0 |
| Relaxation + rounding | `-1.066392` | exact optimum | 0 |
| Relaxation (lower bound) | `-1.120144` | — | ≤ optimum |
| Method A / B / C best feasible `C0` | `-1.066392` | exact optimum | 0 |
| P(optimal): A / B / C | `0.0066 / 0.16 / 0.0007` | exact enumeration | — |
| Beginner penalty threshold `λ*` | `0.109661226` | exhaustive | — |
| Intermediate penalty threshold `λ*` | `0.14936848561939664` | exhaustive (3 code paths) | — |
| Transpiled depth (FakeBrisbane) | `36008` | logical `8782` | `4.10×` |
| Noisy feasibility (1024 shots) | `0.207` | ideal `1.000` | `-0.793` |
| Trotter TV (circuit vs exact) | `0.5827` | exact operator | implementation gap, not noise |
| TV(exact ideal, noisy) | `0.8936` | — | — |

Key plots live in `outputs/phase2/figures/`, `outputs/phase3/figures/` and
`outputs/phase4/figures/`; `portfolio_qaoa.ipynb` reproduces the table numbers.

## How we checked our results

- Every quantum number is checked against an **exact / classical reference**:
  exhaustive enumeration for `n=6` (20) and `n=12` (924), plus independent code
  paths (QUBO vs direct cost, Pauli diagonal vs direct, three evaluators for `λ*`).
- The fast simulators are validated against the explicit Qiskit circuits
  (`fidelity ≥ 1 − 1e-12`, `|ΔE| ≤ 1e-13`).
- **Noiseless vs noisy:** the Phase 4 ideal reference is the noiseless
  statevector of the *same* gate-based circuit that is transpiled and run under
  `FakeBrisbane` noise, so the comparison is apples-to-apples.
- **No real hardware was used**; `FakeBrisbane` is a device model.
- Full `unittest` suite: **127/127 pass** (`exit_code = 0`); see
  `outputs/phase4/validation_report.json`.

## Hardware runs

none — no IBM QPU job was submitted. Phase 4 uses the `FakeBrisbane` device
*model* via `AerSimulator.from_backend(...)`.

## Limitations

- Single synthetic instance; exhaustive search is the true reference, so no
  quantum-advantage claim is possible here.
- Method A is deep (transpiled depth `36008`); the custom weight-K state
  preparation dominates (92.7% of depth) and noise erases the signal.
- `FakeBrisbane` is a noise model, not measured device behaviour.
- Not all COBYLA multistart runs converge; success counts are recorded per start.

## How to run

- Python version: **3.13** (tested on 3.13.14).
- Install: `pip install -r requirements.txt` (qiskit 2.5.2, qiskit-aer 0.17.2,
  qiskit-ibm-runtime 0.50.0, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2).
- **Notebook (recommended):** open `portfolio_qaoa.ipynb` and Run All — it
  reproduces the reported numbers in about a minute and needs no network.
- **Full test suite:** `.venv\Scripts\python.exe -m unittest discover -s tests -t .`
- **Regenerate experiments:** `run_phase1.py`, `run_phase2.py`, `run_phase3.py`
  (+ `run_phase3_validation.py`, `run_phase3_followup.py`), `run_phase4.py`.
  The Phase 4 noisy run takes ~9 minutes; the others are a few minutes.

## Credits

- The Track 2 challenge and tutorial this work builds on are adapted from the
  **BasQ Qiskit Fall Fest** hackathon: tutorials by **Benjamin Tirado** (text
  CC BY 4.0, code Apache 2.0); challenge prompt by Angel Rodriguez, Benjamin
  Tirado and Miguel Angel Jimenez. Citation: *B. Tirado, Combinatorial
  Optimization with QAOA: Max-Cut, tutorial notebook, BasQ Qiskit Fall Fest, 2026.*
- All model, circuit, simulation and analysis code in this repository was written
  by Team axiom.
- **AI tools used:** the opencode AI coding assistant was used to draft and debug
  the Python modules, tests, notebook and slides. Every reported number is
  produced by running the code in this repository; nothing is fabricated.

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, pinned deps, smoke test | complete |
| 1 | Problem model, operators, exact benchmark | complete (49 tests) |
| 2 | Standard penalized QAOA baseline, Beginner instance | complete (82 tests) |
| 3 | Three quantum methods vs classical baselines, Intermediate instance | complete (104 tests) |
| 4 | Hardware awareness & noisy simulation (FakeBrisbane, Aer) | complete (127 tests) |

**Headline result: no quantum advantage was demonstrated.** Classical greedy and
simulated annealing both reach the exact optimum `-1.0663920923466492` with zero
gap in under 2 s, as does relaxation + rounding; the three quantum methods match
that solution quality but do not beat it.

Everything is **simulator-only** — no QPU or IBM Runtime jobs were submitted.

> **Note on environment metadata.** `outputs/phase2/*.json` originally recorded
> package versions that conflicted with the pinned environment (`qiskit_aer` /
> `qiskit_algorithms` "unavailable"; older numpy/scipy/matplotlib). Those
> environment fields were corrected from a fresh probe of the actual interpreter.
> The original values are preserved under
> `package_versions_as_recorded_originally`, and **no measured result was changed
> and Phase 2 was not rerun**.

---

# Detailed phase-by-phase report

## Phase 1 status: complete (model + validation, no optimization yet)

Phase 1 establishes the trusted classical and operator representation of the
cardinality-constrained portfolio problem

    C(x) = q x^T Sigma x - mu^T x + lambda (sum_i x_i - K)^2,   x in {0,1}^n

with feasible set F = { x : sum_i x_i = K }.

## Files

| File | Purpose |
|---|---|
| `phase1_core.py` | Instance generation, objectives, QUBO, QUBO->Pauli map, XY ring mixer, uniform fixed-weight state, exact benchmark |
| `run_phase1.py` | Writes `outputs/phase1/*.json` and runs the test suite |
| `tests/` | `unittest` suite (49 tests) |
| `outputs/phase1/` | Machine-readable results and validation report |
| `phase0_smoke_test.py` | Phase 0 environment smoke test (unchanged) |

## Conventions

- Asset `i` <-> qubit `i`; state index `s = sum_i x_i 2^i` (qubit 0 = LSB).
- Qiskit Pauli strings: rightmost character is qubit 0.
- Operator diagonal entry at index `s` equals `cost_penalized(index_to_assets(s, n))`.

## How to run

```powershell
# from the project root, with the project interpreter
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .   # 49 tests
.\.venv\Scripts\python.exe run_phase1.py                        # writes outputs/phase1/*.json
```

## Confirmed results (Phase 1)

- Instances: Beginner `n=6, K=3` (20 feasible portfolios); Intermediate
  `n=12, K=6` (924 feasible portfolios). Seed `2026`, `q=0.5`.
- Exact optimum (unpenalized): Beginner **-0.503014690** at assets `[1, 2, 5]`;
  Intermediate **-1.066392092** at assets `[1, 5, 6, 8, 10, 11]`.
- QUBO vs direct max abs error: Beginner `4.44e-15`, Intermediate `3.55e-14`.
- Pauli-operator diagonal vs direct max abs error: Beginner `2.66e-15`,
  Intermediate `1.24e-14`.
- XY ring mixer: Hermitian, `[H_M, sum_i Z_i] = 0` (norm 0.0), preserves Hamming
  weight; weight-3 and weight-6 subspace dimensions 20 and 924.
- Uniform fixed-weight state: Beginner 20 nonzero amplitudes, Intermediate 924;
  norm error <= `2.9e-15`.

## Known limitations

- `DickeStatePreparation` is removed in Qiskit 2.x; the uniform fixed-weight
  state is built as an explicit amplitude vector via `StatePreparation`.
- The XY ring mixer uses the literal periodic expression; for `n=2` the single
  edge is listed twice and the effective coupling doubles (documented).
- Synthetic benchmark, not a market-data model.

Package versions are pinned in `requirements.txt` (qiskit 2.5.2, qiskit-aer
0.17.2, qiskit-ibm-runtime 0.50.0, qiskit-algorithms 0.4.0).

## Phase 2 status: complete (standard penalized QAOA baseline, Beginner)

Standard transverse-field QAOA on the Beginner instance (`n=6, K=3, q=0.5`,
seed `2026`): cost Hamiltonian `H_C(lambda)` from Phase 1, mixer
`H_M = sum_i X_i`, initial state `|+>^n`, and per repetition (chronological)
cost evolution `e^{-i gamma H_C}` then mixer `e^{-i beta H_M}`.

- Parameter vector convention: `theta = [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]`
  (this is the explicit circuit's convention; `QAOAAnsatz` is `[beta..., gamma...]`
  and is only used for cross-validation).
- Optimizer: `scipy.optimize.minimize` COBYLA, `rhobeg=1.0`, `maxiter=1500`
  (raised from 250 after all p=3/p=4 starts hit the cap without converging).
- Multistart: 8 independently seeded starts per setting; all starts recorded.
- Sampling: 4096 shots drawn by a seeded multinomial from the exact statevector
  distribution of the validated Qiskit circuit (NOT hardware measurement).
- Energy inner loop uses an exact fast simulator (diagonal cost phases + X
  rotations) validated against the Qiskit circuit (max energy error `5.3e-15`,
  min fidelity `0.999999999999997`).

### Measured results (depth study, `lambda=1.0`)

| p | best `<H_C>` | best feasible `C0` | feas. rate (s/e) | P(opt) (s/e) | score (s/e) | best nfev | wall |
|---|---|---|---|---|---|---|---|
| 1 | 0.010951 | -0.503015 | 0.675 / 0.668 | 0.0354 / 0.0323 | 0.561 / 0.558 | 52 | 1.0 s |
| 2 | -0.262452 | -0.503015 | 0.929 / 0.929 | 0.0486 / 0.0479 | 0.560 / 0.561 | 202 | 37 s |
| 3 | -0.251756 | -0.503015 | 0.895 / 0.895 | 0.0498 / 0.0507 | 0.555 / 0.562 | 1500 | 67 s |
| 4 | -0.352906 | -0.503015 | 0.984 / 0.985 | 0.0488 / 0.0477 | 0.560 / 0.559 | 1500 | 82 s |

(`s` = sampled, `e` = exact statevector; feasible score as defined below.)

### Measured results (penalty study, `p=2`)

| lambda | best `<H_C>` | feas. rate (exact) | P(opt) exact | best feasible `C0` | exact ground feasible |
|---|---|---|---|---|---|
| 0.05 | -0.491589 | 0.085 | 0.0414 | -0.503015 | no |
| 0.10 | -0.416412 | 0.360 | 0.0596 | -0.503015 | no |
| 0.25 | -0.344607 | 0.762 | 0.0441 | -0.503015 | yes |
| 0.50 | -0.331515 | 0.917 | 0.0488 | -0.503015 | yes |
| 1.00 | -0.229389 | 0.919 | 0.0461 | -0.503015 | yes |
| 2.00 | -0.285127 | 0.981 | 0.0401 | -0.503015 | yes |
| 5.00 | 0.221651 | 0.948 | 0.0425 | -0.503015 | yes |

- Exact penalty threshold derived from enumeration: **`lambda* = 0.109661226`**
  (smallest `lambda` making the feasible optimum a global penalized minimum).
- Standard score: `S = (C_worst - E[C0 | feasible]) / (C_worst - C_best)`,
  with `C_best=-0.503014690`, `C_worst=-0.221971172` (null when no feasible
  samples or zero denominator).

### Phase 2 files / reproduction

| File | Purpose |
|---|---|
| `phase2_qaoa.py` | Mixer/cost operators, explicit QAOA circuit, fast exact simulator, COBYLA multistart, sampling, metrics, exact penalty analysis |
| `phase2_experiments.py` | Depth/penalty study orchestration and figures |
| `run_phase2.py` | Runs experiments, writes `outputs/phase2/*.json`, runs tests |
| `tests/test_phase2_*.py` | 33 Phase 2 tests (circuit, energy, sampling, metrics, penalty) |
| `outputs/phase2/` | `depth_results.json`, `penalty_results.json`, `experiment_summary.json`, `validation_report.json`, `figures/` |

```powershell
.\.venv\Scripts\python.exe run_phase2.py        # regeneration + 82 tests
```

### Phase 2 limitations

- One synthetic 6-asset instance; no quantum-advantage claim, and exhaustive
  search is the true reference here.
- Not all multistart COBYLA runs converge (e.g. p=3, p=4: 3/8 succeeded);
  success fractions and termination status are recorded per start.
- Sampled metrics are multinomial draws from the exact statevector
  distribution, not hardware measurements.

## Phase 3 status: complete (three quantum methods vs classical baselines, Intermediate)

Phase 3 runs the Intermediate instance (`n=12, K=6`, seed `2026`, `q=0.5`)
end to end: three quantum variational methods, two classical baselines, a
continuous relaxation, and a cross-method comparison. All work is simulator
only — no QPU / IBM runtime jobs were submitted.

### Methods

| | Method | Mixer | Initial state | Penalty | Simulation in the optimisation loop |
|---|---|---|---|---|---|
| **A** | Constraint-preserving XY-QAOA | `H_M = 1/2 sum(X_i X_j + Y_i Y_j)` on a periodic ring | uniform weight-K state (924 amplitudes) | none — the mixer preserves `sum x_i = K`, so every sample is feasible | exact `WeightSubspaceSimulator` over the 924-dim weight-K subspace |
| **B** | Warm-start QAOA | `sum_i X_i` | `prod_i Ry(2 arcsin sqrt(c*))`, `c*` = continuous relaxation point | `lambda = 1.0` | `FullSpaceSimulator` over 4096 amplitudes |
| **C** | Penalised standard QAOA | `sum_i X_i` | `|+>^12` | `lambda = 1.0` | `FullSpaceSimulator` over 4096 amplitudes |

All three optimise `theta = [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]` with
`scipy.optimize.minimize` COBYLA, `rhobeg=1.0`, `maxiter=1500`.

Settings: depths `p in {1,2,3}` · 5 starts per setting (Method A records 6 for
`p=2,3`) · base seed `2026` (warm start seed `31337`) · 4096 shots drawn by a
seeded multinomial from the exact statevector — **not** hardware measurement.

### Classical baselines and relaxation

| Method | Result | Gap to optimum | Wall |
|---|---|---|---|
| Greedy forward selection | `-1.066392` at `[1,5,6,8,10,11]` | **0 (exact optimum)** | 0.81 ms |
| Feasible-swap simulated annealing | `-1.066392` at `[1,5,6,8,10,11]`, mean over 20 restarts `-1.066392` | **0 (exact optimum)** | 1.34 s (40 000 evals) |
| Continuous relaxation (SLSQP) | `-1.120144`, constraint residual `+0.0770` | `-0.053751` (lower bound) | — |
| Relaxation + rounding/repair | `-1.066392`, feasible | **0 (exact optimum)** | — |

### Measured results — Method A (XY-QAOA, no penalty)

| p | best `<H_C>` | feas. rate (exact) | P(opt) exact | score exact | starts succeeded | wall |
|---|---|---|---|---|---|---|
| 1 | -0.690626 | 1.000 | 0.0005 | 0.5197 | 3/5 | 62 s |
| 2 | -0.700973 | 1.000 | 0.0006 | 0.5329 | 0/6 | 193 s |
| 3 | -0.727339 | 1.000 | 0.0066 | 0.5666 | 0/6 | 466 s |

Feasibility is 1.000 at every depth by construction (the mixer lives inside the
weight-6 subspace), and every depth's 4096-shot best feasible `C0` equals
`-1.066392`.

### Measured results — Method B (warm start) and Method C (penalised)

| Method | p | best `<H_C(lambda)>` | feas. rate (exact) | P(opt) exact | score exact | starts succeeded | wall |
|---|---|---|---|---|---|---|---|
| B | 1 | -0.096931 | 0.422 | **0.1037** | 0.9438 | 5/5 | 0.5 s |
| B | 2 | -0.358740 | 0.536 | **0.1494** | 0.9483 | 5/5 | 6.3 s |
| B | 3 | -0.562913 | 0.691 | **0.1609** | 0.9420 | 3/5 | 869 s |
| C | 1 | 1.092275 | 0.211 | 0.0002 | 0.4893 | 5/5 | 0.6 s |
| C | 2 | -0.050614 | 0.548 | 0.0005 | 0.4671 | 5/5 | 3.3 s |
| C | 3 | -0.150121 | 0.606 | 0.0007 | 0.4542 | 5/5 | 6.3 s |

B and C have no automatic feasibility guarantee, so their sample space mixes
infeasible portfolios; the `score` column is conditioned on feasible samples
only. Method B's warm start is clearly the better of the two — it delivers
over 200x the optimal-solution probability of Method C at every depth.

### Penalty diagnostic (Method C, `p=2`)

| lambda | feas. rate (exact) | P(opt) exact | E[C0] exact |
|---|---|---|---|
| 0.10 | 0.308 | 0.0027 | -0.778299 |
| 1.00 | 0.815 | 0.0007 | -0.665803 |
| 2.00 | 0.493 | 0.0008 | -0.657854 |

### Exact penalty threshold for the Intermediate instance

Derived by exhaustive enumeration of all 4096 assignments, computed twice by
independent code paths and confirmed by a third independent evaluator:

- **`lambda* = 0.14936848561939664`** (smallest `lambda` making the feasible
  optimum a global penalized minimum).
- Argmax assignment: index **3442**, assets **`[1, 4, 5, 6, 8, 10, 11]`**,
  cardinality **7**, `C0 = -1.2157605779660459` (the K+1 portfolio with the
  most negative unpenalized cost). At `lambda*` exactly, this portfolio and the
  feasible optimum are **tied** global minimizers; just above, the feasible
  optimum is the unique minimizer; just below, the infeasible portfolio wins.
- Only weight >= 7 constrains `lambda*`: every cardinality <= 5 portfolio has
  `C0 > C_best`, so no penalty is needed to suppress them.
- Compare with the Beginner threshold from Phase 2, `lambda* = 0.109661226`.
  Phase 3 ran at `lambda = 1.0`, comfortably above both — so the low
  feasibility of Methods B and C is a QAOA-dynamics effect, **not** a penalty
  that is too weak.

Written to `outputs/phase3/intermediate_penalty_threshold.json`.

### Verdict

**No quantum advantage was demonstrated.** Classical greedy *and*
simulated annealing both reach the exact optimum `-1.0663920923466492` at
assets `[1, 5, 6, 8, 10, 11]` with zero gap — as does relaxation rounding.
On the quantum side the best feasible objective sampled by A, B and C is also
`-1.066392`, so all three methods tie the classical baselines on solution
quality; they differ only in how much probability they put on it (B `0.16`,
A `0.0066`, C `0.0007`) and in cost (B/C ~1 s, A ~720 s over all depths).
The quantum methods are therefore reported on solution quality, feasibility,
optimal-solution probability and cost — not on advantage.

### Validation (regenerated 2026-10-13, fresh measurements)

| Check | Result |
|---|---|
| Full test suite (child process, real exit code) | **104/104 pass**, `exit_code=0`, 3.3 s |
| Method A subspace vs full Qiskit circuit, `n=12, p=1` | fidelity `1.0000000000000004`, `|dE| = 5.22e-15`, infeasible mass `3.97e-13` (46 s) |
| Method A subspace vs full Qiskit circuit, `n=12, p=3` | fidelity `1.0000000000000000`, `|dE| = 0.00e+00`, infeasible mass `3.99e-13` (134 s) |
| Method B `FullSpaceSimulator` vs Qiskit circuit, `n=6, p in {1,2,3}` | min fidelity `0.9999999999999964`, max `|dE| = 1.60e-14` |
| Method C `FullSpaceSimulator` vs Qiskit circuit, `n=6, p in {1,2,3}` | min fidelity `0.9999999999999876`, max `|dE| = 1.00e-13` |
| **Method B vs Qiskit, `n=12, p=3`** (follow-up) | fidelity `1.0000000000000022`, `|dE| = 8.88e-16`, TV `1.24e-15`, norm err `2.22e-16`/`1.33e-15`, 1.9 s |
| **Method C vs Qiskit, `n=12, p=3`** (follow-up) | fidelity `0.9999999999999982`, `|dE| = 8.33e-17`, TV `2.13e-15`, norm err `0.0`/`1.78e-15`, 1.5 s |
| `decompose` faithfulness at `n=6` (measured, not assumed) | min fidelity `0.9999999999999787`, max `|dE| = 1.60e-14` |

Both `n=12` Method A checks completed inside the 600 s wall-clock timeout, and
both `n=12` Method B/C follow-up circuits completed in under 2 s, so **every
check above was freshly measured** — no value in
`outputs/phase3/validation_report.json` or
`outputs/phase3/final_followup_validation.json` is carried forward from an
earlier run.

Note on the `n=12` Method B/C follow-up: the first attempt timed out at the
genuine 600 s wall-clock limit because `quantum_info.Statevector` exponentiates
the opaque `PauliEvolutionGate` generator matrix instead of using the gate's
own definition (measured: ~14 s per cost layer, >386 s per X-mixer layer at
`n=12`). The corrected run expands the circuit to its own Qiskit definition
via `QuantumCircuit.decompose(reps=4)` before simulation — unitary-preserving,
and its faithfulness is measured directly at `n=6` in the last row above. The
superseded timeouts are preserved under `prior_attempts` in the follow-up
JSON.

### Phase 3 files / reproduction

| File | Purpose |
|---|---|
| `phase3_methods.py` | Weight-subspace and full-space simulators, XY-QAOA circuit, warm start, relaxation + rounding, greedy, SA |
| `phase3_experiments.py` | `run_method_a/b/c`, penalty diagnostic, metrics helpers, figures |
| `run_phase3.py` | Full pipeline: experiments, comparison, figures, validation, tests |
| `run_phase3_validation.py` | **Targeted** validation regeneration only (does not rerun experiments) |
| `run_phase3_followup.py` | Final follow-up: Intermediate penalty threshold + `n=12` Methods B/C vs Qiskit |
| `tests/test_phase3_*.py` | 22 Phase 3 tests (XY-QAOA, warm start, classical, metrics) |
| `outputs/phase3/` | 7 experiment JSONs + `intermediate_penalty_threshold.json` + `final_followup_validation.json` + `validation_report.json` + `figures/` (7 PNGs) |

```powershell
.\.venv\Scripts\python.exe run_phase3.py                 # full regeneration + 104 tests
.\.venv\Scripts\python.exe -u run_phase3_validation.py   # validation/report only
.\.venv\Scripts\python.exe -u run_phase3_followup.py     # final follow-up checks
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .   # 104 tests
```

### Phase 3 limitations

- Single synthetic 12-asset instance; exhaustive search over 924 feasible
  portfolios is the true reference, and both classical baselines already hit it.
- Not all multistart COBYLA runs converge: Method A reports `0/6` succeeded for
  `p=2,3` (every start hit the `maxiter=1500` evaluation cap without reporting
  success). Energies are near-insensitive to the cap (compare `maxiter=500` and
  `1500`), but the "success" counts should be read with that in mind.
- Sampled metrics are multinomial draws from the exact statevector
  distribution, not hardware measurements; a 4096-shot sample makes "best
  feasible" a weak discriminator (all methods tie), so `P(opt)` and the
  feasible-conditioned mean are the primary metrics.
- Method A's subspace simulation is O(924) per layer but its runtime is
  dominated by Qiskit's `StatePreparation` and `PauliEvolutionGate` synthesis
  outside the loop; Method B at `p=3` took 869 s for 5 starts.
- Circuit depths reported are logical, not transpiled.

## Phase 4 status: complete (hardware awareness & noisy simulation, FakeBrisbane)

Phase 4 takes the Method A circuit selected from the Phase 3 artifact
(`p=3`, `best_theta = [5.648123, 6.380744, -2.624672, 4.277143, 2.480479,
2.379861]`, artifact `sha256 = 33b0b95f…`), reconstructs it as a gate-based
circuit, transpiles it for the IBM `FakeBrisbane` target, and compares a
noiseless statevector reference with a `FakeBrisbane`-noisy Aer run of the
**same** circuit. Everything is **simulator only** — no QPU or IBM Runtime job
was submitted. No Phase 3 artifact was modified or re-optimized.

### Backend and subgraph

| Property | Value |
|---|---|
| Backend | `fake_brisbane` (127 qubits, 144 coupling edges) |
| Basis gates | `ecr, id, rz, sx, x` |
| 12-qubit subgraph | layout `[0..11]`, a path with **11 induced edges** (the densest connected 12-node region of the heavy-hex map) |
| Why a subgraph | the full 127-qubit transpile has no connectivity advantage worth its width and produced unstable `Statevector`/`Aer` runs in the reference scratch code |

### Experiment A — logical vs transpiled resources

| Circuit | depth | size | two-qubit gates | ops |
|---|---|---|---|---|
| logical (opaque) | 7 | 7 | — (opaque) | `PauliEvolution:6, state_preparation:1` |
| logical (decomposed, `reps=4`) | 8782 | 9600 | 4623 (`cx`) | `u:4689, cx:4623, p:288` |
| state prep only (transpiled) | 33392 | 67719 | 9429 (`ecr`) | `rz:33933, sx:22747, ecr:9429, x:1610` |
| transpiled (12-qubit subgraph) | 36008 | 76430 | 10554 (`ecr`) | `rz:38282, sx:25932, ecr:10554, x:1662` |
| transpiled (full 127-qubit device) | 34092 | 63925 | 8813 (`ecr`) | `rz:31895, sx:21504, ecr:8813, x:1713` |

- Transpiler: `optimization_level=3`, `seed_transpiler=2026`, with a fixed
  `initial_layout`; **no explicit `swap` instruction survives** — routing cost
  shows up as extra `ecr` gates and depth.
- Routing need: **2378 of 4623** logical two-qubit pairs are non-adjacent under
  the chosen layout (51.4%).
- Overheads vs the decomposed logical circuit: subgraph **4.10x depth / 2.28x
  two-qubit gates**; full device **3.88x depth / 1.91x two-qubit gates**.
- The custom fixed-weight `StatePreparation` dominates: **89.3%** of two-qubit
  gates and **92.7%** of depth of the transpiled subgraph circuit.

### Experiment B — ideal vs noisy

The primary **ideal** reference is the exact `Statevector` of the same
decomposed circuit that is transpiled and run under noise (no measurement), so
ideal vs noisy is apples-to-apples. Both the ideal and the noisy runs use 4
decimal-identical circuit, layout and gate decomposition.

| Metric | ideal (circuit statevector) | noisy (FakeBrisbane, 1024 shots) |
|---|---|---|
| feasibility rate | 1.000000000000 | 0.20703125 (212/1024) |
| P(optimal) | 0.0021148 | 0.0000 (0/1024) |
| best feasible `C0` | -1.066392 | -1.026446 |
| mean `C0` \| feasible | -0.658622 | -0.648193 |
| feasible score | 0.47875 | 0.46542 |

- Noisy run: `AerSimulator.from_backend(FakeBrisbane())`, `shots=1024`,
  `seed_simulator=2026`, **549.7 s** inside a genuine 1800 s child-process
  timeout (genuine timeout; no value is fabricated). 908 distinct outcomes of
  4096 possible, largest count 3 — the distribution is essentially uniform, i.e.
  the signal is destroyed by noise.
- Comparison: `TV(exact ideal, noisy) = 0.8936`; equal-shot sampled-vs-sampled
  `TV = 0.9072` (both at 1024 shots); feasibility-rate drop `0.793`.
- The noisy feasibility `0.207` is close to the uniform-subspace fraction
  `924/4096 = 0.2256`, the expected value for a fully depolarized register: at
  depth `36008` with `10554` `ecr` gates the circuit is far past the noise
  threshold, so useful signal cannot survive.

### Trotterization / implementation gap (not hardware noise)

The Method A **opaque** `PauliEvolutionGate` reference used in Phase 3 is exact
(fidelity `1.0` against the weighted-subspace simulator). But a **gate-based,
hardware-transpilable** circuit must decompose that gate, and Qiskit's
decomposition of the non-commuting XY-ring mixer is a first-order Trotter
product:

| Quantity | Value |
|---|---|
| circuit-vs-exact-subspace TV (n=12, p=3) | 0.5827 |
| circuit-vs-exact-subspace max `|dP|` | 0.02337 |
| circuit-vs-exact-subspace fidelity (feasible subspace) | 0.5149 |
| P(opt): circuit vs Phase 3 exact | 0.0021148 vs 0.0066357 |
| `E[C0]`: circuit vs Phase 3 exact | -0.658622 vs -0.727339 |

This gap is an **implementation** effect (decomposition of the mixer), not
hardware noise, and is reported separately from the ideal-vs-noisy comparison.
It is a real property of running the method on gate-based hardware that the
Phase 3 operator-level validation could not see.

### Validation and preservation

| Check | Result |
|---|---|
| Full test suite (child process, real exit code) | **127/127 pass**, `exit_code=0`, 75.1 s |
| Measurement decoding (known optimum bitstring) | `110101100010` → index `3426` ✅ |
| Transpiled subgraph: all 2q gates on coupling edges | ✅ `0` offending |
| Transpiled subgraph: all ops in target basis | ✅ |
| Ideal circuit constraint preserved | infeasible mass `0.0`, state norm `1.0` |
| Noisy run | completed in 549.7 s (timeout 1800 s) |
| Phase 3 result artifacts (10 JSON + 7 figures) | **17/17 unchanged** (sha256 + mtime) |
| Phase 3 validation report blocks | preserved; this README section is additive |
| QPU / IBM Runtime jobs | **none** |

Git `HEAD` is unchanged at `ee6dc5a` (Phase 0 commit); no commit, push, remote,
or rewrite was performed.

### Phase 4 files / reproduction

| File | Purpose |
|---|---|
| `phase4_hardware.py` | Method A circuit selection/provenance, subgraph selection, transpile + resource analysis, ideal + noisy simulation, comparison, figures |
| `run_phase4.py` | Orchestration: writes `outputs/phase4/*.json`, figures and runs the test suite |
| `tests/test_phase4_hardware.py` | 23 fast deterministic Phase 4 tests (decoding, selection, transpile validity, metrics, ideal/trotter structure) |
| `outputs/phase4/` | `circuit_resources.json`, `ideal_results.json`, `noisy_results.json`, `comparison_summary.json`, `validation_report.json`, `figures/` (2 PNGs) |

```powershell
.\.venv\Scripts\python.exe -u run_phase4.py    # full pipeline + 127 tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .   # 127 tests
```

### Phase 4 limitations

- FakeBrisbane is a **device model**, not real hardware; the noisy results are
  simulated noise, not measured device behaviour.
- The noisy study used a 12-qubit subgraph with a fixed layout (the choice is
  documented); the full 127-qubit device transpires with *fewer* two-qubit gates
  here, so the subgraph is not the best possible connectivity.
- `AerSimulator` was left on `method="automatic"`; the effective method is
  recorded in `noisy_results.json` but not pinned.
- The Trotterization gap means the Phase 3 Method A `P(opt)` is an
  operator-level value; the hardware-realizable circuit's `P(opt)` is ~3x lower.
- As everywhere in this project, **no quantum advantage** is claimed.
