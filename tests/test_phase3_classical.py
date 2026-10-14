"""Phase 3 tests: classical baselines (greedy forward selection, SA)."""

import unittest

import numpy as np

import phase1_core as core
import phase3_methods as p3


class TestClassicalBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst6 = core.generate_instance(6, 3)
        cls.bench6 = core.exact_benchmark(6, 3, cls.inst6.mu, cls.inst6.Sigma,
                                          cls.inst6.q)

    def test_greedy_is_feasible_and_consistent(self):
        g = p3.greedy_forward_selection(self.inst6)
        x = np.asarray(g["assignment"])
        self.assertEqual(int(x.sum()), self.inst6.K)
        self.assertEqual(sorted(g["asset_indices"]), sorted(np.where(x)[0].tolist()))
        self.assertAlmostEqual(
            g["objective"],
            core.cost_unpenalized(x, self.inst6.mu, self.inst6.Sigma,
                                  self.inst6.q), places=12)

    def test_greedy_is_deterministic(self):
        a = p3.greedy_forward_selection(self.inst6)
        b = p3.greedy_forward_selection(self.inst6)
        self.assertEqual(a["asset_indices"], b["asset_indices"])
        self.assertEqual(a["objective"], b["objective"])

    def test_greedy_reaches_known_optimum_n6(self):
        g = p3.greedy_forward_selection(self.inst6)
        self.assertAlmostEqual(g["objective"],
                              self.bench6["best_feasible_objective"], places=9)

    def test_sa_is_feasible_deterministic_and_bounded(self):
        a = p3.run_simulated_annealing(self.inst6, 8, 400, 0.2, 0.002,
                                       base_seed=101)
        b = p3.run_simulated_annealing(self.inst6, 8, 400, 0.2, 0.002,
                                       base_seed=101)
        self.assertEqual(a["best_cost"], b["best_cost"])
        self.assertTrue(a["best_assignment"] == b["best_assignment"])
        self.assertEqual(a["total_evaluations"], 8 * 400)
        for r in a["restarts"]:
            self.assertTrue(r["is_feasible"])
            self.assertLessEqual(r["best_cost"], r["final_cost"] + 1e-12)
        self.assertLessEqual(a["best_cost"], a["mean_best_cost"] + 1e-12)

    def test_sa_best_is_at_least_as_good_as_greedy_n6(self):
        g = p3.greedy_forward_selection(self.inst6)
        sa = p3.run_simulated_annealing(self.inst6, 10, 500, 0.3, 0.002,
                                        base_seed=7)
        self.assertLessEqual(sa["best_cost"], g["objective"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
