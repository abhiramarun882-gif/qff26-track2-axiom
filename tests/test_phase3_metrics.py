"""Phase 3 tests: metric helpers used by the experiment comparison."""

import unittest

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa
import phase3_experiments as exp


class TestPhase3Metrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.bench = core.exact_benchmark(6, 3, cls.inst.mu, cls.inst.Sigma,
                                         cls.inst.q)
        cls.ctx = qaoa.MetricContext.build(cls.inst, 0.0, cls.bench)

    def _opt_index(self):
        return core.assets_to_index(self.bench["optimal_assignment"])

    def test_expected_unpenalized_on_optimum(self):
        idx = self._opt_index()
        val = exp.expected_unpenalized({idx: 1.0}, 1.0, self.ctx)
        self.assertAlmostEqual(val, self.ctx.C_best, places=12)

    def test_distribution_metrics_adds_expected_c0(self):
        idx = self._opt_index()
        m = exp._distribution_metrics({idx: 1.0}, 1.0, self.ctx)
        self.assertIn("expected_unpenalized_cost", m)
        self.assertAlmostEqual(m["expected_unpenalized_cost"], self.ctx.C_best,
                              places=12)
        self.assertAlmostEqual(m["feasibility_rate"], 1.0, places=12)

    def test_sample_block_sums_to_shots_and_is_deterministic(self):
        probs = np.zeros(64)
        probs[self._opt_index()] = 1.0
        a = exp._sample_block(probs, self.ctx, 512, seed=42)
        b = exp._sample_block(probs, self.ctx, 512, seed=42)
        self.assertEqual(a["counts_checked_sum"], 512)
        self.assertEqual(a["sampled_metrics"], b["sampled_metrics"])
        self.assertAlmostEqual(
            a["exact_statevector_metrics"]["optimal_solution_probability"],
            1.0, places=12)

    def test_starts_summary(self):
        starts = [
            qaoa.StartResult(0, [0.0], [0.1], -1.0, 12, True, 0, "ok", 0.01),
            qaoa.StartResult(1, [0.0], [0.2], -0.5, 20, True, 0, "ok", 0.02),
            qaoa.StartResult(2, [0.0], [np.nan], float("nan"), 5, False, -1,
                             "fail", 0.005),
        ]
        s = exp._starts_summary(starts)
        self.assertEqual(s["num_starts"], 3)
        self.assertEqual(s["num_successful"], 2)
        self.assertEqual(s["num_valid_finite"], 2)
        self.assertAlmostEqual(s["energy_min"], -1.0, places=12)

    def test_warmstart_theta_extends_depth(self):
        prev = [0.1, 0.2, 0.3, 0.4]  # reps=2 -> reps=3
        out = exp._warmstart_theta(prev, 3)
        self.assertEqual(len(out), 6)
        self.assertAlmostEqual(out[2], 0.2, places=12)  # repeated last gamma
        self.assertAlmostEqual(out[5], 0.4, places=12)  # repeated last beta


if __name__ == "__main__":
    unittest.main()
