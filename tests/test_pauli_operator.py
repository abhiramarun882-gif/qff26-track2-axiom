"""Category C: QUBO -> Pauli operator mapping (independent diagonal check)."""

import unittest

import numpy as np

import phase1_core as core


class TestPauliOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = core.generate_instance(6, 3)
        cls.i = core.generate_instance(12, 6)

    def test_diagonal_matches_direct_all_beginner(self):
        lam = 1.0
        op, c_I, h, J = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, lam)
        diag = core.operator_diagonal(op)
        max_err = 0.0
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            direct = core.cost_penalized(x, self.b.mu, self.b.Sigma, 3, lam, self.b.q)
            max_err = max(max_err, abs(diag[s] - direct))
        self.assertLessEqual(max_err, core.MATRIX_ATOL, f"max_err={max_err}")

    def test_diagonal_matches_direct_intermediate(self):
        lam = 0.75
        op, c_I, h, J = core.qubo_to_pauli(self.i.mu, self.i.Sigma, self.i.q, 6, lam)
        diag = core.operator_diagonal(op)
        max_err = 0.0
        for x in core.feasible_assignments(12, 6):
            s = core.assets_to_index(x)
            direct = core.cost_penalized(x, self.i.mu, self.i.Sigma, 6, lam, self.i.q)
            max_err = max(max_err, abs(diag[s] - direct))
        rng = np.random.default_rng(23)
        tested = 0
        while tested < 500:
            s = int(rng.integers(0, 2 ** 12))
            x = core.index_to_assets(s, 12)
            if int(x.sum()) == 6:
                continue
            direct = core.cost_penalized(x, self.i.mu, self.i.Sigma, 6, lam, self.i.q)
            max_err = max(max_err, abs(diag[s] - direct))
            tested += 1
        self.assertLessEqual(max_err, core.MATRIX_ATOL, f"max_err={max_err}")

    def test_operator_hermitian(self):
        op, *_ = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, 1.0)
        diff = (op - op.adjoint()).to_matrix(sparse=True)
        self.assertLessEqual(abs(diff).max() if diff.nnz else 0.0, core.HERMITICITY_ATOL)

    def test_identity_coefficient_present(self):
        op, c_I, h, J = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, 1.0)
        labels = list(op.paulis.to_labels())
        self.assertIn("I" * 6, labels)
        idx = labels.index("I" * 6)
        self.assertAlmostEqual(complex(op.coeffs[idx]).real, c_I, places=12)

    def test_term_structure(self):
        n = 6
        op, c_I, h, J = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, 1.0)
        labels = list(op.paulis.to_labels())
        # exactly one identity, n single-Z, C(n,2) ZZ (all coefficients nonzero here)
        singles = [l for l in labels if l.count("Z") == 1]
        doubles = [l for l in labels if l.count("Z") == 2]
        self.assertEqual(sum(1 for l in labels if l == "I" * n), 1)
        self.assertEqual(len(singles), n)
        self.assertEqual(len(doubles), n * (n - 1) // 2)
        # no X or Y in a cost operator
        self.assertTrue(all(("X" not in l and "Y" not in l) for l in labels))

    def test_operator_dimension(self):
        op6, *_ = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, 1.0)
        op12, *_ = core.qubo_to_pauli(self.i.mu, self.i.Sigma, self.i.q, 6, 1.0)
        self.assertEqual(op6.num_qubits, 6)
        self.assertEqual(op12.num_qubits, 12)

    def test_coefficients_match_formula(self):
        lam = 0.9
        c, a, B = core.qubo_coefficients(self.b.mu, self.b.Sigma, self.b.q, 3, lam)
        op, c_I, h, J = core.qubo_to_pauli(self.b.mu, self.b.Sigma, self.b.q, 3, lam)
        n = 6
        expected_cI = c + 0.5 * a.sum() + 0.25 * B.sum()
        self.assertAlmostEqual(c_I, expected_cI, places=12)
        for i in range(n):
            s = sum(B[min(i, j), max(i, j)] for j in range(n) if j != i)
            self.assertAlmostEqual(h[i], -0.5 * a[i] - 0.25 * s, places=12)
        for i in range(n):
            for j in range(i + 1, n):
                self.assertAlmostEqual(J[i, j], B[i, j] / 4.0, places=12)


if __name__ == "__main__":
    unittest.main()
