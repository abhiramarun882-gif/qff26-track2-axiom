"""Category D: exact classical benchmark."""

import unittest

import numpy as np

import phase1_core as core


class TestExactBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = core.generate_instance(6, 3)
        cls.i = core.generate_instance(12, 6)
        cls.rb = core.exact_benchmark(6, 3, cls.b.mu, cls.b.Sigma, cls.b.q)
        cls.ri = core.exact_benchmark(12, 6, cls.i.mu, cls.i.Sigma, cls.i.q)

    def test_beginner_count(self):
        self.assertEqual(self.rb["feasible_count"], 20)
        self.assertTrue(self.rb["feasible_count_ok"])

    def test_intermediate_count(self):
        self.assertEqual(self.ri["feasible_count"], 924)
        self.assertTrue(self.ri["feasible_count_ok"])

    def test_beginner_optimum_matches_independent(self):
        self.assertTrue(self.rb["independent_agreement"])
        self.assertAlmostEqual(self.rb["optimal_unpenalized_cost"],
                               self.rb["independent_optimal_cost"], places=9)
        self.assertEqual(self.rb["optimal_asset_indices"],
                         self.rb["independent_optimal_indices"])

    def test_intermediate_optimum_matches_independent(self):
        self.assertTrue(self.ri["independent_agreement"])
        self.assertAlmostEqual(self.ri["optimal_unpenalized_cost"],
                               self.ri["independent_optimal_cost"], places=9)
        # 12 choose 6 = 924 independent feasible states must agree
        self.assertEqual(self.ri["independent_feasible_count"], 924)

    def test_optimal_assignment_is_feasible_and_optimal(self):
        for r, inst in ((self.rb, self.b), (self.ri, self.i)):
            x = r["optimal_assignment"]
            self.assertTrue(core.is_feasible(x, inst.K))
            val = core.cost_unpenalized(x, inst.mu, inst.Sigma, inst.q)
            self.assertAlmostEqual(val, r["optimal_unpenalized_cost"], places=9)
            # no feasible portfolio beats it
            objs = [core.cost_unpenalized(f, inst.mu, inst.Sigma, inst.q)
                    for f in core.feasible_assignments(inst.n, inst.K)]
            self.assertGreaterEqual(min(objs), r["optimal_unpenalized_cost"] - 1e-9)

    def test_best_and_worst_bracket_distribution(self):
        for r in (self.rb, self.ri):
            self.assertLessEqual(r["best_feasible_objective"],
                                 r["worst_feasible_objective"])
            self.assertAlmostEqual(r["best_feasible_objective"],
                                   r["sorted_feasible_objectives"][0], places=9)
            self.assertAlmostEqual(r["worst_feasible_objective"],
                                   r["sorted_feasible_objectives"][-1], places=9)

    def test_distribution_length(self):
        self.assertEqual(len(self.rb["sorted_feasible_objectives"]), 20)
        self.assertEqual(len(self.ri["sorted_feasible_objectives"]), 924)

    def test_determinism(self):
        r2 = core.exact_benchmark(6, 3, self.b.mu, self.b.Sigma, self.b.q)
        self.assertEqual(r2["optimal_unpenalized_cost"],
                         self.rb["optimal_unpenalized_cost"])
        self.assertEqual(r2["optimal_asset_indices"], self.rb["optimal_asset_indices"])

    def test_full_space_penalized_bounds(self):
        for r in (self.rb, self.ri):
            self.assertLessEqual(r["full_space_penalized_min"],
                                 r["full_space_penalized_max"])


if __name__ == "__main__":
    unittest.main()
