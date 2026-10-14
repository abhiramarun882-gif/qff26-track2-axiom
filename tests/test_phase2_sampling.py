"""Phase 2 tests: statevector probabilities and shot sampling."""

import unittest

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa


class TestSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.lam = 1.0
        cls.cost = qaoa.cost_operator(cls.inst, cls.lam)
        cls.qc = qaoa.build_qaoa_circuit(6, 2, cls.cost, qaoa.build_x_mixer(6))
        cls.theta = [0.4, 1.1, 0.7, 1.9]
        cls.probs = qaoa.exact_probabilities(cls.qc, cls.theta)

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(float(self.probs.sum()), 1.0, places=10)

    def test_counts_sum_to_shots(self):
        counts = qaoa.sample_counts_exact(self.probs, qaoa.SHOTS, seed=1234)
        self.assertEqual(sum(counts.values()), qaoa.SHOTS)

    def test_sampling_is_deterministic(self):
        c1 = qaoa.sample_counts_exact(self.probs, qaoa.SHOTS, seed=7)
        c2 = qaoa.sample_counts_exact(self.probs, qaoa.SHOTS, seed=7)
        self.assertEqual(c1, c2)

    def test_counts_decode_with_phase1_convention(self):
        counts = qaoa.sample_counts_exact(self.probs, qaoa.SHOTS, seed=99)
        for idx in counts:
            self.assertTrue(0 <= idx < 2 ** 6)
            x = core.index_to_assets(idx, 6)
            self.assertEqual(core.assets_to_index(x), idx)

    def test_optimal_state_is_identified(self):
        bench = core.exact_benchmark(6, 3, self.inst.mu, self.inst.Sigma,
                                     self.inst.q)
        ctx = qaoa.MetricContext.build(self.inst, self.lam, bench)
        opt_idx = core.assets_to_index(bench["optimal_assignment"])
        probs = np.zeros(2 ** 6)
        probs[opt_idx] = 1.0
        counts = qaoa.sample_counts_exact(probs, 100, seed=3)
        m = qaoa.metrics_from_counts(counts, ctx)
        self.assertAlmostEqual(m["optimal_solution_probability"], 1.0, places=12)
        self.assertAlmostEqual(m["feasibility_rate"], 1.0, places=12)
        self.assertAlmostEqual(m["best_feasible_unpenalized_cost"],
                               ctx.C_best, places=12)

    def test_feasibility_determination_from_counts(self):
        bench = core.exact_benchmark(6, 3, self.inst.mu, self.inst.Sigma,
                                     self.inst.q)
        ctx = qaoa.MetricContext.build(self.inst, self.lam, bench)
        # index 0 is the all-zero (infeasible) state.
        m = qaoa.metrics_from_counts({0: 50}, ctx)
        self.assertEqual(m["feasibility_rate"], 0.0)
        self.assertIsNone(m["feasible_normalized_score"])


if __name__ == "__main__":
    unittest.main()
