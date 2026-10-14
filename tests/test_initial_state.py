"""Category F: uniform fixed-weight (Dicke-like) initial state."""

import math
import unittest

import numpy as np

import phase1_core as core


class TestInitialState(unittest.TestCase):
    def test_beginner_state(self):
        diag = core.weight_state_support_counts(6, 3)
        self.assertEqual(diag["dimension"], 64)
        self.assertEqual(diag["expected_nonzero"], 20)
        self.assertEqual(diag["nonzero_amplitudes"], 20)
        self.assertEqual(diag["support_count"], 20)
        self.assertTrue(diag["support_is_weight_K"])
        self.assertLessEqual(diag["norm_error"], core.NORM_ATOL)

    def test_intermediate_state(self):
        diag = core.weight_state_support_counts(12, 6)
        self.assertEqual(diag["dimension"], 4096)
        self.assertEqual(diag["expected_nonzero"], 924)
        self.assertEqual(diag["nonzero_amplitudes"], 924)
        self.assertEqual(diag["support_count"], 924)
        self.assertTrue(diag["support_is_weight_K"])
        self.assertLessEqual(diag["norm_error"], core.NORM_ATOL)

    def test_amplitudes_equal_and_named_state(self):
        vec = core.uniform_weight_state(6, 3)
        nz = vec[np.abs(vec) > 1e-15]
        self.assertEqual(nz.shape[0], 20)
        self.assertTrue(np.allclose(nz, 1.0 / math.sqrt(20)))

    def test_edge_cases(self):
        # K = 0 -> only |00...0>; K = n -> only |11...1>
        v0 = core.uniform_weight_state(4, 0)
        np.testing.assert_allclose(v0, np.eye(1, 16, 0).ravel())
        vn = core.uniform_weight_state(4, 4)
        np.testing.assert_allclose(vn, np.eye(1, 16, 15).ravel())

    def test_invalid_K_rejected(self):
        with self.assertRaises(ValueError):
            core.uniform_weight_state(6, 7)
        with self.assertRaises(ValueError):
            core.uniform_weight_state(6, -1)


if __name__ == "__main__":
    unittest.main()
