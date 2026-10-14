"""Category B: classical objectives and QUBO expansion."""

import unittest

import numpy as np

import phase1_core as core


class TestCostOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = core.generate_instance(6, 3)
        cls.i = core.generate_instance(12, 6)

    def _random_assignments(self, n, count, seed):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=(count, n))

    def test_vectorized_equals_loop_unpenalized(self):
        for x in self._random_assignments(6, 200, 1):
            v1 = core.cost_unpenalized(x, self.b.mu, self.b.Sigma, self.b.q)
            v2 = core.cost_penalized_loop(x, self.b.mu, self.b.Sigma, 3, 0.0, self.b.q)
            self.assertAlmostEqual(v1, v2, places=9)

    def test_vectorized_equals_loop_penalized(self):
        for lam in (0.0, 0.5, 2.0):
            for x in self._random_assignments(6, 100, 2):
                v1 = core.cost_penalized(x, self.b.mu, self.b.Sigma, 3, lam, self.b.q)
                v2 = core.cost_penalized_loop(x, self.b.mu, self.b.Sigma, 3, lam, self.b.q)
                self.assertAlmostEqual(v1, v2, places=9)

    def test_penalty_separation(self):
        for x in self._random_assignments(6, 50, 3):
            lam = 1.3
            c0 = core.cost_unpenalized(x, self.b.mu, self.b.Sigma, self.b.q)
            c1 = core.cost_penalized(x, self.b.mu, self.b.Sigma, 3, lam, self.b.q)
            expected_pen = lam * (int(np.sum(x)) - 3) ** 2
            self.assertAlmostEqual(c1 - c0, expected_pen, places=9)

    def test_invalid_vectors_rejected(self):
        with self.assertRaises(ValueError):
            core.cost_unpenalized([0, 1, 2, 0, 0, 0], self.b.mu, self.b.Sigma)
        with self.assertRaises(ValueError):
            core.cost_penalized([0.4, 1, 0, 0, 0, 0], self.b.mu, self.b.Sigma, 3, 1.0)
        with self.assertRaises(ValueError):
            core.cost_unpenalized([0, 1, 0], self.b.mu, self.b.Sigma)

    def test_qubo_matches_direct_all_beginner(self):
        c, a, B = core.qubo_coefficients(self.b.mu, self.b.Sigma, self.b.q, 3, 1.0)
        max_err = 0.0
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            direct = core.cost_penalized(x, self.b.mu, self.b.Sigma, 3, 1.0, self.b.q)
            qubo = core.cost_from_qubo(x, c, a, B)
            max_err = max(max_err, abs(direct - qubo))
        self.assertLessEqual(max_err, core.OBJECTIVE_ATOL, f"max_err={max_err}")

    def test_qubo_matches_intermediate_feasible_and_sample(self):
        lam = 0.75
        c, a, B = core.qubo_coefficients(self.i.mu, self.i.Sigma, self.i.q, 6, lam)
        max_err = 0.0
        for x in core.feasible_assignments(12, 6):
            direct = core.cost_penalized(x, self.i.mu, self.i.Sigma, 6, lam, self.i.q)
            qubo = core.cost_from_qubo(x, c, a, B)
            max_err = max(max_err, abs(direct - qubo))
        rng = np.random.default_rng(11)
        tested = 0
        while tested < 500:
            s = int(rng.integers(0, 2 ** 12))
            x = core.index_to_assets(s, 12)
            if int(x.sum()) == 6:
                continue
            direct = core.cost_penalized(x, self.i.mu, self.i.Sigma, 6, lam, self.i.q)
            qubo = core.cost_from_qubo(x, c, a, B)
            max_err = max(max_err, abs(direct - qubo))
            tested += 1
        self.assertLessEqual(max_err, core.OBJECTIVE_ATOL, f"max_err={max_err}")

    def test_qubo_lambda_does_not_change_feasible_ranking(self):
        # On the feasible set the penalty is identically zero, so the
        # unpenalized objective is lambda-independent.
        feasible = core.feasible_assignments(6, 3)
        objs0 = [core.cost_unpenalized(x, self.b.mu, self.b.Sigma, self.b.q)
                 for x in feasible]
        for lam in (0.0, 0.5, 3.0):
            c, a, B = core.qubo_coefficients(self.b.mu, self.b.Sigma, self.b.q, 3, lam)
            for x, o0 in zip(feasible, objs0):
                pen = core.cost_penalized(x, self.b.mu, self.b.Sigma, 3, lam, self.b.q)
                self.assertAlmostEqual(pen, o0, places=9)
                self.assertAlmostEqual(core.cost_from_qubo(x, c, a, B), o0, places=9)

    def test_feasibility_checker(self):
        self.assertTrue(core.is_feasible([1, 1, 1, 0, 0, 0], 3))
        self.assertFalse(core.is_feasible([1, 1, 1, 1, 0, 0], 3))
        self.assertFalse(core.is_feasible([1, 0.5, 1, 0, 0, 0], 3))


if __name__ == "__main__":
    unittest.main()
