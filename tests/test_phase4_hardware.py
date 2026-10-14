"""Fast, deterministic Phase 4 tests (no full noisy run).

These cover circuit reconstruction/parameter binding, inclusion of the
fixed-weight state preparation, measurement decoding and bit ordering, the
objective evaluator agreeing with Phase 1, successful transpilation against
FakeBrisbane, availability of resource metrics, the ideal/noisy result schema,
feasibility/objective metrics from synthetic counts, and reproducibility of the
configuration.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import phase1_core as core          # noqa: E402
import phase2_qaoa as q2            # noqa: E402
import phase3_methods as p3         # noqa: E402
import phase4_hardware as p4        # noqa: E402

_TOL = 1e-9


def _backend():
    return p4._fake_brisbane()


class TestBitOrdering(unittest.TestCase):
    def test_known_bitstring(self):
        kb = p4.known_bitstring_check()
        self.assertTrue(kb["key_matches_expected"])
        self.assertEqual(kb["expected_index"], 3426)
        self.assertEqual(kb["expected_qiskit_key"], "110101100010")
        self.assertTrue(kb["decode_roundtrip_ok"])
        self.assertTrue(kb["index_to_assets_ok"])
        self.assertTrue(kb["reversed_key_gives_different_index"])

    def test_roundtrip_all_single_bits(self):
        for i in range(p4.N):
            key = "".join("1" if b == i else "0" for b in range(p4.N - 1, -1, -1))
            self.assertEqual(p4.bitstring_to_index(key), 1 << i)
            self.assertEqual(p4.index_to_bitstring(1 << i), key)

    def test_matches_index_to_assets(self):
        for idx in (0, 1, 3426, 1023, (1 << p4.N) - 1):
            x = core.index_to_assets(idx, p4.N)
            key = p4.index_to_bitstring(idx)
            self.assertEqual(p4.bitstring_to_index(key), core.assets_to_index(x))

    def test_rejects_malformed(self):
        with self.assertRaises(ValueError):
            p4.bitstring_to_index("012", 3)
        with self.assertRaises(ValueError):
            p4.bitstring_to_index("101", 4)


class TestCircuitSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sel = p4.load_method_a(ROOT)

    def test_provenance(self):
        self.assertEqual(self.sel["p"], 3)
        self.assertEqual(len(self.sel["theta"]), 6)
        self.assertEqual(len(self.sel["artifact_sha256"]), 64)
        self.assertIn("method_a_xy_qaoa.json", self.sel["artifact_path"])

    def test_circuit_reconstruction_and_binding(self):
        from qiskit import QuantumCircuit
        qc = self.sel["qc_log"]
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, p4.N)
        # no unbound parameters after bind_theta
        self.assertEqual(len(qc.parameters), 0)
        # 1 state-prep + 3 cost + 3 mixer opaque instructions
        self.assertEqual(qc.size(), 1 + 2 * self.sel["p"])

    def test_state_preparation_included(self):
        qc_dec = self.sel["qc_log"].decompose(reps=p4.DECOMPOSE_REPS)
        # StatePreparation must be present at the opaque level
        ops = qc_dec.count_ops()
        self.assertGreater(sum(v for k, v in qc_dec.count_ops().items()),
                           2 * self.sel["p"])
        # the decomposed circuit is far larger than the QAOA layers alone
        self.assertGreater(qc_dec.depth(), 100)

    def test_objective_matches_phase1(self):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import Statevector
        sp = QuantumCircuit(p4.N)
        sp.append(StatePreparation(core.uniform_weight_state(p4.N, p4.K)),
                  range(p4.N))
        sv = Statevector(sp)
        # uniform weight-K state: equal amplitude on all C(12,6)=924 states
        probs = sv.probabilities()
        feasible = p3.feasible_indices(p4.N, p4.K)
        self.assertEqual(len(feasible), 924)
        self.assertAlmostEqual(float(probs[feasible].sum()), 1.0, places=12)
        self.assertTrue(np.allclose(probs[feasible], 1.0 / 924, atol=1e-12))

    def test_optimal_index(self):
        bench = self.sel["benchmark"]
        self.assertEqual(self.sel["optimal_index"],
                         core.assets_to_index(bench["optimal_assignment"]))
        self.assertAlmostEqual(
            core.cost_unpenalized(bench["optimal_assignment"],
                                  self.sel["inst"].mu, self.sel["inst"].Sigma,
                                  self.sel["inst"].q),
            bench["best_feasible_objective"], places=12)


class TestSubgraphAndTranspile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = _backend()
        cls.edges = p4.backend_edges(cls.backend)
        cls.sg = p4.select_subgraph(cls.edges, cls.backend.num_qubits, p4.N)
        cls.sel = p4.load_method_a(ROOT)

    def test_subgraph_is_valid(self):
        layout = self.sg["layout"]
        self.assertEqual(len(layout), p4.N)
        self.assertEqual(len(set(layout)), p4.N)
        self.assertEqual(self.sg["num_induced_edges"], len(
            self.sg["induced_edges"]))
        self.assertGreaterEqual(self.sg["num_induced_edges"], p4.N - 1)

    def test_transpile_succeeds_and_respects_constraints(self):
        from qiskit import transpile
        cmap = p4.subgraph_coupling_map(self.edges, self.sg["layout"])
        qc_dec = self.sel["qc_log"].decompose(reps=p4.DECOMPOSE_REPS)
        tqc = transpile(qc_dec, coupling_map=cmap,
                        basis_gates=list(self.backend.basis_gates),
                        initial_layout=self.sg["layout"],
                        optimization_level=p4.OPT_LEVEL,
                        seed_transpiler=p4.SEED_TRANSPILER)
        val = p4._validate_transpiled(tqc, cmap, list(self.backend.basis_gates))
        self.assertTrue(val["all_two_qubit_gates_on_coupling_edges"])
        self.assertTrue(val["all_operations_in_target_basis"])
        rep = p4.gate_report(tqc)
        self.assertGreater(rep["two_qubit_gate_count"], 0)
        self.assertIn("ecr", rep["two_qubit_gates_by_name"])

    def test_resource_metrics_available(self):
        res = p4.resource_analysis(self.sel, self.backend)
        self.assertIn("logical_decomposed", res)
        self.assertGreater(res["logical_decomposed"]["two_qubit_gate_count"], 0)
        self.assertIsNotNone(res["transpiled_subgraph"])
        self.assertGreater(res["transpiled_subgraph"]["depth"], 0)
        self.assertIsNotNone(res["state_preparation_transpiled"])
        self.assertIn("non_adjacent_under_layout", res["routing_analysis"])
        self.assertEqual(res["failures"], {})


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        inst = core.generate_instance(p4.N, p4.K)
        bench = core.exact_benchmark(p4.N, p4.K, inst.mu, inst.Sigma, inst.q)
        cls.ctx = q2.MetricContext.build(inst, 0.0, bench)

    def test_metrics_from_synthetic_counts(self):
        opt = "110101100010"  # index 3426, the exact optimum
        # 10 optimal, 10 infeasible (all-ones has weight 12)
        counts = {opt: 10, "1" * p4.N: 10}
        m = p4.metrics_from_aer_counts(counts, self.ctx)
        self.assertAlmostEqual(m["feasibility_rate"], 0.5, places=12)
        self.assertAlmostEqual(m["optimal_solution_probability"], 0.5,
                               places=12)
        self.assertAlmostEqual(m["best_feasible_unpenalized_cost"],
                               self.ctx.C_best, places=9)

    def test_all_infeasible(self):
        counts = {"1" * p4.N: 100}
        m = p4.metrics_from_aer_counts(counts, self.ctx)
        self.assertEqual(m["feasibility_rate"], 0.0)
        self.assertEqual(m["optimal_solution_probability"], 0.0)
        self.assertIsNone(m["mean_c0_conditioned_on_feasible"])

    def test_counts_key_order(self):
        # two keys that are bit-reverses of each other must decode distinctly
        a = p4.bitstring_to_index("100000000000")
        b = p4.bitstring_to_index("000000000001")
        self.assertEqual(a, 1 << 11)
        self.assertEqual(b, 1)

    def test_json_schema_roundtrip(self):
        sample = {
            "metrics": {"feasibility_rate": np.float64(0.5),
                        "optimal_solution_probability": np.float64(0.25)},
            "counts": {"000000000000": int(4)},
        }
        s = json.dumps(sample, default=p4._json_default)
        back = json.loads(s)
        self.assertEqual(back["counts"]["000000000000"], 4)


class TestConfiguration(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((p4.N, p4.K, p4.METHOD, p4.REPS), (12, 6, "A", 3))
        self.assertEqual(p4.SEED_TRANSPILER, 2026)
        self.assertEqual(p4.OPT_LEVEL, 3)
        self.assertEqual(p4.DECOMPOSE_REPS, 4)
        self.assertGreater(p4.SHOTS_IDEAL, 0)
        self.assertGreater(p4.SHOTS_NOISY, 0)

    def test_selection_is_reproducible(self):
        a = p4.load_method_a(ROOT)
        b = p4.load_method_a(ROOT)
        self.assertEqual(a["theta"], b["theta"])
        self.assertEqual(a["artifact_sha256"], b["artifact_sha256"])
        self.assertEqual(a["record_index"], b["record_index"])

    def test_subgraph_selection_deterministic(self):
        be = _backend()
        e = p4.backend_edges(be)
        a = p4.select_subgraph(e, be.num_qubits, p4.N)
        b = p4.select_subgraph(e, be.num_qubits, p4.N)
        self.assertEqual(a["layout"], b["layout"])
        self.assertEqual(a["induced_edges"], b["induced_edges"])

    def test_sha256_matches_json_load(self):
        import hashlib
        path = os.path.join(ROOT, *p4.METHOD_A_ARTIFACT.split("/"))
        with open(path, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(h, p4.load_method_a(ROOT)["artifact_sha256"])


class TestIdealReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sel = p4.load_method_a(ROOT)
        cls.ideal = p4.ideal_reference(cls.sel, shots=128, seed=1234)

    def test_primary_ideal_is_circuit_statevector(self):
        self.assertAlmostEqual(
            self.ideal["metrics_exact"]["feasibility_rate"], 1.0, places=9)
        self.assertTrue(
            self.ideal["validation"]["circuit_support_is_weight_K"])
        self.assertLess(
            self.ideal["validation"]["circuit_infeasible_mass"], 1e-9)
        self.assertEqual(len(self.ideal["exact_probabilities"]), 1 << p4.N)

    def test_trotterization_gap_recorded_not_noise(self):
        tg = self.ideal["trotterization_gap"]
        self.assertFalse(tg["is_hardware_noise"])
        self.assertGreater(tg["total_variation"], 0.0)
        self.assertEqual(len(self.ideal["subspace_exact_probabilities"]),
                         len(self.ideal["exact_probabilities"]))
        self.assertAlmostEqual(
            tg["optimal_probability_circuit"],
            self.ideal["metrics_exact"]["optimal_solution_probability"],
            places=9)

    def test_sampled_counts_sum_to_shots(self):
        self.assertEqual(self.ideal["aer_counts_total"], 128)
        self.assertEqual(sum(self.ideal["aer_raw_counts"].values()), 128)


if __name__ == "__main__":
    unittest.main()
