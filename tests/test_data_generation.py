"""Category A: deterministic instance generation."""

import unittest

import numpy as np

import phase1_core as core


class TestDataGeneration(unittest.TestCase):
    def test_same_seed_is_identical(self):
        a = core.generate_instance(6, 3, seed=2026)
        b = core.generate_instance(6, 3, seed=2026)
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.Sigma, b.Sigma)

    def test_different_seed_differs(self):
        a = core.generate_instance(6, 3, seed=2026)
        b = core.generate_instance(6, 3, seed=7)
        self.assertFalse(np.allclose(a.mu, b.mu))

    def test_shapes(self):
        inst = core.generate_instance(6, 3)
        self.assertEqual(inst.mu.shape, (6,))
        self.assertEqual(inst.Sigma.shape, (6, 6))
        inst12 = core.generate_instance(12, 6)
        self.assertEqual(inst12.Sigma.shape, (12, 12))

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            core.generate_instance(0, 0)
        with self.assertRaises(ValueError):
            core.generate_instance(6, 7)
        with self.assertRaises(ValueError):
            core.generate_instance(6, -1)
        with self.assertRaises(ValueError):
            core.generate_instance(6, 3, q=-0.1)
        with self.assertRaises(ValueError):
            core.generate_instance(6, 3, q=np.nan)
        with self.assertRaises(ValueError):
            core.generate_instance(6, 3, q=np.inf)
        with self.assertRaises(TypeError):
            core.generate_instance(6.0, 3)

    def test_boundary_cases(self):
        # K = 0 and K = n are valid instances.
        z = core.generate_instance(5, 0)
        self.assertEqual(z.K, 0)
        full = core.generate_instance(5, 5)
        self.assertEqual(full.K, 5)
        # n = 1
        one = core.generate_instance(1, 1)
        self.assertEqual(one.Sigma.shape, (1, 1))

    def test_symmetry_and_psd(self):
        for (n, K) in [(6, 3), (12, 6)]:
            diag = core.check_instance_properties(core.generate_instance(n, K))
            self.assertTrue(diag["is_symmetric"], diag)
            self.assertTrue(diag["is_psd"], diag)
            self.assertLessEqual(diag["max_asymmetry"], core.SYMMETRY_ATOL)

    def test_beginner_and_intermediate_are_independent(self):
        b = core.generate_instance(6, 3, seed=2026)
        i = core.generate_instance(12, 6, seed=2026)
        # The covariance is NOT the Beginner block extended to 12x12.
        self.assertFalse(np.allclose(b.Sigma, i.Sigma[:6, :6]))
        # The intermediate instance follows the documented n=12 draw order.
        rng = np.random.default_rng(2026)
        mu = rng.uniform(0.05, 0.25, size=12)
        A = rng.normal(0.0, 0.1, size=(12, 12))
        np.testing.assert_allclose(i.mu, mu, atol=0.0)
        np.testing.assert_allclose(i.Sigma, A @ A.T, atol=0.0)
        # NOTE: the first 6 uniform draws coincide between n=6 and n=12 because
        # both consume the same RNG stream prefix. That is expected and is not
        # evidence of the Intermediate instance being built by extension.
        np.testing.assert_allclose(i.mu[:6], b.mu, atol=0.0)

    def test_to_dict_round_trip(self):
        inst = core.generate_instance(6, 3)
        d = inst.to_dict()
        mu = np.array(d["mu"])
        Sigma = np.array(d["Sigma"])
        np.testing.assert_allclose(mu, inst.mu, atol=0.0)
        np.testing.assert_allclose(Sigma, inst.Sigma, atol=0.0)
        self.assertEqual(d["seed"], 2026)
        self.assertEqual(d["K"], 3)


if __name__ == "__main__":
    unittest.main()
