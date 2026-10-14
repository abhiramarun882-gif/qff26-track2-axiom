"""Bit-ordering convention test, independent of the QUBO coefficient path.

Builds a 2-qubit operator by hand with distinguishable single-qubit
coefficients and checks the diagonal against the declared convention
(qubit 0 = least-significant bit). This pins down the asset<->qubit<->index
mapping without reusing any reversal helper from the QUBO mapping.
"""

import unittest

import numpy as np
from qiskit.quantum_info import SparsePauliOp

import phase1_core as core


class TestBitOrdering(unittest.TestCase):
    def test_two_qubit_distinguishable_coefficients(self):
        h0, h1, J = 1.0, 2.0, 0.5
        # In Qiskit strings the rightmost char is qubit 0.
        op = SparsePauliOp.from_list([
            ("IZ", h0),   # Z on qubit 0
            ("ZI", h1),   # Z on qubit 1
            ("ZZ", J),
        ])
        diag = core.operator_diagonal(op)

        # Manual: z_i = +1 if x_i == 0 else -1, x_i = (s >> i) & 1
        def expected(s):
            x0 = (s >> 0) & 1
            x1 = (s >> 1) & 1
            z0 = 1 - 2 * x0
            z1 = 1 - 2 * x1
            return h0 * z0 + h1 * z1 + J * z0 * z1

        for s in range(4):
            self.assertAlmostEqual(diag[s], expected(s), places=12)

        # Explicit spot checks: index 1 -> x=[1,0] -> z=[-1,+1]
        self.assertAlmostEqual(diag[1], -h0 + h1 - J, places=12)
        self.assertAlmostEqual(diag[2], h0 - h1 - J, places=12)
        self.assertAlmostEqual(diag[0], h0 + h1 + J, places=12)
        self.assertAlmostEqual(diag[3], -h0 - h1 + J, places=12)

    def test_index_assignment_round_trip(self):
        for s in range(2 ** 5):
            x = core.index_to_assets(s, 5)
            self.assertEqual(core.assets_to_index(x), s)

    def test_index0_is_all_zero_and_index_is_little_endian(self):
        np.testing.assert_array_equal(core.index_to_assets(0, 4), [0, 0, 0, 0])
        # index 1 sets only asset 0
        np.testing.assert_array_equal(core.index_to_assets(1, 4), [1, 0, 0, 0])
        # index 8 sets only asset 3
        np.testing.assert_array_equal(core.index_to_assets(8, 4), [0, 0, 0, 1])

    def test_cost_operator_uses_asset_index_equals_qubit_index(self):
        # A single-qubit Z-cost on one asset: coefficient on asset i must show
        # up at bit position i of the operator diagonal.
        n = 4
        mu = np.array([0.1, 0.1, 0.1, 0.1])
        Sigma = np.zeros((n, n))
        # only a linear mu term -> cost = -mu^T x => Z term h_i = +mu_i/2
        op, c_I, h, J = core.qubo_to_pauli(mu, Sigma, q=0.0, K=0, lam=0.0)
        diag = core.operator_diagonal(op)
        for s in range(2 ** n):
            x = core.index_to_assets(s, n)
            self.assertAlmostEqual(diag[s], -float(np.dot(mu, x)), places=12)


if __name__ == "__main__":
    unittest.main()
