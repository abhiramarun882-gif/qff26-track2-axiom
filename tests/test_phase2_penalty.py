"""Phase 2 tests: exact penalty analysis against independent enumeration."""

import math
import unittest

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa


class TestPenaltyAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.lambdas = list(qaoa.PENALTIES)
        cls.analysis = qaoa.exact_penalty_analysis(cls.inst, cls.lambdas)

    def _bruteforce(self, lam):
        """Independent loop enumeration of the penalized minimum."""
        best = math.inf
        best_idx = None
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            val = core.cost_penalized_loop(x, self.inst.mu, self.inst.Sigma,
                                           self.inst.K, lam, self.inst.q)
            if val < best - 1e-12:
                best = val
                best_idx = s
        return best, best_idx

    def test_global_minima_agree_with_bruteforce(self):
        for lam in self.lambdas:
            rec = self.analysis["per_lambda"][f"{lam:g}"]
            best, best_idx = self._bruteforce(lam)
            self.assertAlmostEqual(rec["global_penalized_min"], best, places=9)
            # the reported minimizer must attain the reported minimum
            x = core.index_to_assets(rec["global_minimizer_index"], 6)
            attained = core.cost_penalized_loop(x, self.inst.mu, self.inst.Sigma,
                                                self.inst.K, lam, self.inst.q)
            self.assertAlmostEqual(attained, rec["global_penalized_min"], places=9)

    def test_feasibility_status_from_ground_states(self):
        X = core.all_assignments(6)
        for lam in self.lambdas:
            rec = self.analysis["per_lambda"][f"{lam:g}"]
            idx = rec["global_minimizer_index"]
            self.assertEqual(rec["global_minimizer_feasible"],
                             bool(X[idx].sum() == 3))

    def test_threshold_matches_direct_enumeration(self):
        bench = core.exact_benchmark(6, 3, self.inst.mu, self.inst.Sigma,
                                     self.inst.q)
        c_best = bench["best_feasible_objective"]
        ratios = []
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            k = int(x.sum())
            if k == 3:
                continue
            c0 = core.cost_unpenalized(x, self.inst.mu, self.inst.Sigma,
                                       self.inst.q)
            gap = c_best - c0
            if gap > 0:
                ratios.append(gap / (k - 3) ** 2)
        expected = max(ratios) if ratios else 0.0
        self.assertAlmostEqual(self.analysis["penalty_threshold"], expected,
                               places=9)
        self.assertTrue(self.analysis["penalty_threshold"] > 0)

    def test_at_or_above_threshold_feasible_optimum_is_global(self):
        thr = self.analysis["penalty_threshold"]
        for lam in [thr, thr * 1.5, 5.0]:
            rec = qaoa.exact_penalty_analysis(self.inst, [lam])["per_lambda"][f"{lam:g}"]
            self.assertTrue(rec["feasible_optimum_is_global_min"])
            self.assertFalse(rec["infeasible_beats_feasible"])


if __name__ == "__main__":
    unittest.main()
