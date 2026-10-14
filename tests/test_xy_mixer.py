"""Category E: XY ring mixer properties."""

import math
import unittest

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

import phase1_core as core


class TestXYMixer(unittest.TestCase):
    def test_hermiticity(self):
        for n in (4, 6):
            op = core.build_xy_ring_mixer(n)
            diff = (op - op.adjoint()).to_matrix(sparse=True)
            maxabs = abs(diff).max() if diff.nnz else 0.0
            self.assertLessEqual(maxabs, core.HERMITICITY_ATOL)

    def test_operator_dimension(self):
        for n in (2, 4, 6, 12):
            self.assertEqual(core.build_xy_ring_mixer(n).num_qubits, n)

    def test_not_x_only(self):
        labels = list(core.build_xy_ring_mixer(6).paulis.to_labels())
        self.assertTrue(any("Y" in l for l in labels))
        self.assertTrue(any("X" in l for l in labels))

    def test_excitation_conservation_commutator(self):
        for n in (4, 6):
            hm = core.build_xy_ring_mixer(n)
            zsum = core.excitation_number_operator(n)
            M = Operator(hm).data
            Z = Operator(zsum).data
            comm = M @ Z - Z @ M
            self.assertLessEqual(np.linalg.norm(comm), core.COMMUTATOR_ATOL)

    def test_hamming_weight_preserved_small(self):
        n = 4
        hm = core.build_xy_ring_mixer(n)
        M = np.asarray(hm.to_matrix(sparse=True).todense())
        for i in range(2 ** n):
            for j in range(2 ** n):
                if abs(M[i, j]) > 1e-12:
                    self.assertEqual(core.hamming_weight(i), core.hamming_weight(j),
                                     f"mixer connects weight {core.hamming_weight(i)} "
                                     f"to {core.hamming_weight(j)} at ({i},{j})")

    def test_periodic_ring_kind_six_has_n_edges(self):
        n = 6
        labels = list(core.build_xy_ring_mixer(n).paulis.to_labels())
        xx = [l for l in labels if l.count("X") == 2]
        yy = [l for l in labels if l.count("Y") == 2]
        self.assertEqual(len(xx), n)
        self.assertEqual(len(yy), n)

    def test_n2_documented_doubling(self):
        # For n=2 the periodic expression lists edge (0,1) twice; after
        # simplify the effective coupling is 0.5 + 0.5 = 1.0 (documented).
        op = core.build_xy_ring_mixer(2)
        coeffs = {label: complex(c) for label, c in zip(op.paulis.to_labels(), op.coeffs)}
        self.assertAlmostEqual(coeffs["XX"].real, 1.0, places=12)
        self.assertAlmostEqual(coeffs["YY"].real, 1.0, places=12)

    def test_subspace_dimensions(self):
        self.assertEqual(math.comb(6, 3), 20)
        self.assertEqual(math.comb(12, 6), 924)


if __name__ == "__main__":
    unittest.main()
