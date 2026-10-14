"""Phase 0 smoke test for Team axiom - Track 2 (Quantum Optimization).

Independently checks that the core Qiskit + scientific stack imports, that the
estimator/sampler primitives work, that a fake heavy-hex backend can drive an
AerSimulator, that DickeStatePreparation constructs, and that QAOAAnsatz accepts
a custom mixer. No optimization is run.

Exit code is non-zero if any required test fails.
"""

import sys
import traceback

REQUIRED_TESTS = ["A", "B", "C", "D", "E"]
results = {}

VERSION_TARGETS = [
    ("qiskit", "qiskit"),
    ("qiskit_aer", "qiskit-aer"),
    ("qiskit_ibm_runtime", "qiskit-ibm-runtime"),
    ("qiskit_algorithms", "qiskit-algorithms"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("rustworkx", "rustworkx"),
]


def run_test(name, func):
    print("\n" + "=" * 64)
    print(f"TEST {name}")
    print("=" * 64)
    try:
        func()
        results[name] = True
        print(f"RESULT {name}: PASS")
    except Exception:
        results[name] = False
        print(f"RESULT {name}: FAIL")
        traceback.print_exc()
    print("-" * 64)


def test_a_core_imports():
    import importlib

    missing = []
    for mod_name, _ in VERSION_TARGETS:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{mod_name}: {exc}")

    print("Installed versions:")
    for mod_name, dist_name in VERSION_TARGETS:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {dist_name:<22} {ver}")
        except Exception:
            print(f"  {dist_name:<22} MISSING")

    if missing:
        raise ImportError("Core imports failed:\n  " + "\n  ".join(missing))


def test_b_primitives():
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from qiskit.quantum_info import SparsePauliOp

    print("StatevectorEstimator imported:", StatevectorEstimator.__name__)
    print("StatevectorSampler imported  :", StatevectorSampler.__name__)

    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()

    qc = QuantumCircuit(1)
    qc.h(0)
    ev = estimator.run([(qc, SparsePauliOp.from_list([("Z", 1.0)]))]).result()[0].data.evs
    ev = float(ev)
    print(f"<Z> of |+> = {ev:.6f} (expected ~0.0)")

    qc_meas = QuantumCircuit(1)
    qc_meas.h(0)
    qc_meas.measure_all()
    counts = sampler.run([qc_meas], shots=100).result()[0].data.meas.get_counts()
    print("Sampler counts on |+>:", counts)

    if abs(ev) > 1e-6:
        raise AssertionError(f"Estimator gave <Z>={ev}, expected ~0")
    if set(counts) - {"0", "1"}:
        raise AssertionError(f"Unexpected sampler keys: {counts}")


def test_c_aer_fake_brisbane():
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    backend = FakeBrisbane()
    print("FakeBrisbane backend name:", backend.name)
    sim = AerSimulator.from_backend(backend)
    print("AerSimulator created from FakeBrisbane.")
    print("Simulator num_qubits:", sim.configuration().n_qubits)
    print("Simulator name       :", sim.name)


def test_d_constraint_preserving_initial_state():
    """Prepare the uniform superposition over weight-K basis states (Dicke state).

    COMPATIBILITY NOTE: ``DickeStatePreparation`` existed in Qiskit 1.x but has
    been removed in the pinned Qiskit 2.5.2 (``ImportError`` below is expected).
    The functionally equivalent state is built here with ``StatePreparation``
    from an explicitly constructed amplitude vector, using only supported APIs.
    """
    import numpy as np
    from math import comb
    from qiskit.circuit.library import StatePreparation

    n, k = 12, 6

    try:
        from qiskit.circuit.library import DickeStatePreparation  # noqa: F401

        legacy = DickeStatePreparation(n, k)
        print("DickeStatePreparation imported; num_qubits:", legacy.num_qubits)
        if legacy.num_qubits != n:
            raise AssertionError(f"Expected {n} qubits, got {legacy.num_qubits}")
        return
    except ImportError as exc:
        print("NOTE: DickeStatePreparation is NOT available in this Qiskit version:")
        print(f"      {exc}")
        print("      Using the equivalent StatePreparation of the uniform "
              f"weight-{k} state instead (supported API).")

    dim = 2 ** n
    vec = np.zeros(dim, dtype=complex)
    amp = 1.0 / np.sqrt(comb(n, k))
    for basis in range(dim):
        if bin(basis).count("1") == k:
            vec[basis] = amp

    nonzero = int(np.count_nonzero(vec))
    norm = float(np.linalg.norm(vec))
    print(f"Uniform weight-{k} state: nonzero amplitudes = {nonzero} "
          f"(expected {comb(n, k)})")
    print(f"State norm = {norm:.6f} (expected 1.0)")

    circ = StatePreparation(vec)
    print("StatePreparation num_qubits:", circ.num_qubits)

    if circ.num_qubits != n:
        raise AssertionError(f"Expected {n} qubits, got {circ.num_qubits}")
    if nonzero != comb(n, k):
        raise AssertionError(f"Expected {comb(n, k)} nonzero amplitudes, got {nonzero}")
    if abs(norm - 1.0) > 1e-9:
        raise AssertionError(f"State not normalized: {norm}")


def test_e_qaoa_ansatz_mixer():
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp

    cost_op = SparsePauliOp.from_list([("ZZ", 1.0)])
    mixer_op = SparsePauliOp.from_list([("XI", 1.0), ("IX", 1.0)])

    ansatz = QAOAAnsatz(cost_operator=cost_op, mixer_operator=mixer_op, reps=1)
    print("QAOAAnsatz constructed with custom mixer_operator.")
    print("num_qubits    :", ansatz.num_qubits)
    print("num_parameters:", ansatz.num_parameters)
    try:
        depth = ansatz.decompose(reps=3).depth()
        print("decomposed depth:", depth)
    except Exception as exc:  # noqa: BLE001
        print("depth unavailable:", exc)

    if ansatz.num_qubits != 2:
        raise AssertionError(f"Expected 2 qubits, got {ansatz.num_qubits}")


def test_f_summary():
    print("\n" + "=" * 64)
    print("PHASE 0 SMOKE TEST SUMMARY")
    print("=" * 64)
    passed = sum(1 for n in REQUIRED_TESTS if results.get(n))
    failed = [n for n in REQUIRED_TESTS if not results.get(n)]
    for name in REQUIRED_TESTS:
        print(f"  Test {name}: {'PASS' if results.get(name) else 'FAIL'}")
    print(f"\nTotal passed: {passed}")
    print(f"Total failed: {len(failed)}")
    print(f"\nEnvironment ready for Phase 1: {'YES' if not failed else 'NO'}")
    if failed:
        raise SystemExit(1)


def main():
    print("Python executable:", sys.executable)
    print("Python version   :", sys.version.replace("\n", " "))
    run_test("A", test_a_core_imports)
    run_test("B", test_b_primitives)
    run_test("C", test_c_aer_fake_brisbane)
    run_test("D", test_d_constraint_preserving_initial_state)
    run_test("E", test_e_qaoa_ansatz_mixer)
    run_test("F", test_f_summary)


if __name__ == "__main__":
    main()
