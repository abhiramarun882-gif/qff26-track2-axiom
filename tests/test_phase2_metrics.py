"""Phase 2 tests: metric definitions (normalised feasible score)."""

import unittest

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.lam = 1.0
        cls.bench = core.exact_benchmark(6, 3, cls.inst.mu, cls.inst.Sigma,
                                         cls.inst.q)
        cls.ctx = qaoa.MetricContext.build(cls.inst, cls.lam, cls.bench)
        cls.opt_idx = core.assets_to_index(cls.bench["optimal_assignment"])
        # worst feasible index
        feasible = core.feasible_assignments(6, 3)
        vals = [core.cost_unpenalized(x, cls.inst.mu, cls.inst.Sigma,
                                      cls.inst.q) for x in feasible]
        worst_row = int(np.argmax(vals))
        cls.worst_idx = core.assets_to_index(feasible[worst_row])

    def test_score_is_one_for_optimal_only_distribution(self):
        m = qaoa.aggregate_metrics({self.opt_idx: 1.0}, 1.0, self.ctx)
        self.assertAlmostEqual(m["feasible_normalized_score"], 1.0, places=12)
        self.assertAlmostEqual(m["feasibility_rate"], 1.0, places=12)

    def test_score_is_zero_for_worst_only_distribution(self):
        m = qaoa.aggregate_metrics({self.worst_idx: 1.0}, 1.0, self.ctx)
        self.assertAlmostEqual(m["feasible_normalized_score"], 0.0, places=12)

    def test_infeasible_samples_excluded_from_conditional_score(self):
        mix = {self.opt_idx: 0.5, 0: 0.25, 63: 0.25}  # 0 and 63 infeasible
        m = qaoa.aggregate_metrics(mix, 1.0, self.ctx)
        self.assertAlmostEqual(m["feasibility_rate"], 0.5, places=12)
        self.assertAlmostEqual(m["feasible_normalized_score"], 1.0, places=12)
        self.assertAlmostEqual(m["mean_c0_conditioned_on_feasible"],
                               self.ctx.C_best, places=12)

    def test_undefined_score_when_no_feasible_samples(self):
        m = qaoa.aggregate_metrics({0: 1.0}, 1.0, self.ctx)
        self.assertIsNone(m["feasible_normalized_score"])
        self.assertIsNone(m["mean_c0_conditioned_on_feasible"])
        self.assertEqual(m["feasibility_rate"], 0.0)

    def test_zero_denominator_reported_as_none(self):
        ctx = qaoa.MetricContext(n=6, K=3, mu=self.ctx.mu, Sigma=self.ctx.Sigma,
                                 q=self.ctx.q, lam=self.lam, C_best=1.0,
                                 C_worst=1.0, optimal_indices=self.ctx.optimal_indices)
        m = qaoa.aggregate_metrics({self.opt_idx: 1.0}, 1.0, ctx)
        self.assertIsNone(m["feasible_normalized_score"])

    def test_weighted_conditional_mean(self):
        # half optimal, half worst -> conditional mean midway
        m = qaoa.aggregate_metrics({self.opt_idx: 0.5, self.worst_idx: 0.5},
                                   1.0, self.ctx)
        expected = 0.5 * (self.ctx.C_best + self.ctx.C_worst)
        self.assertAlmostEqual(m["mean_c0_conditioned_on_feasible"], expected,
                               places=12)
        self.assertAlmostEqual(m["feasible_normalized_score"], 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
