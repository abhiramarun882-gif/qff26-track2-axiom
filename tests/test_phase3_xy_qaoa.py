"""Phase 3 tests: Method A constraint-preserving XY-QAOA (subspace simulator)."""

import math
import unittest

import numpy as np
from qiskit.quantum_info import Statevector

import phase1_core as core
import phase2_qaoa as qaoa
import phase3_methods as p3


class TestXYQAOA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.K = 6, 3
        cls.inst = core.generate_instance(cls.n, cls.K)
        cls.cost0 = core.qubo_to_pauli(cls.inst.mu, cls.inst.Sigma, cls.inst.q,
                                       cls.K, 0.0)[0]
        cls.diag0 = core.operator_diagonal(cls.cost0)
        cls.feas = p3.feasible_indices(cls.n, cls.K)
        cls.rng = np.random.default_rng(2024)

    def test_feasible_indices(self):
        self.assertEqual(self.feas.shape[0], math.comb(self.n, self.K))
        for s in self.feas:
            self.assertEqual(bin(int(s)).count("1"), self.K)

    def test_mixer_block_is_symmetric(self):
        sim = p3.WeightSubspaceSimulator(self.n, self.K, self.diag0, 1)
        m = sim.mixer_matrix
        self.assertEqual(m.shape, (math.comb(self.n, self.K),) * 2)
        self.assertTrue(np.allclose(m, m.T, atol=1e-12))

    def test_statevector_normalised(self):
        sim = p3.WeightSubspaceSimulator(self.n, self.K, self.diag0, 2)
        amp = sim.statevector(self.rng.uniform(0, 6, size=4))
        self.assertAlmostEqual(float(np.sum(np.abs(amp) ** 2)), 1.0, places=12)

    def test_uniform_start_is_uniform_over_feasible(self):
        sim = p3.WeightSubspaceSimulator(self.n, self.K, self.diag0, 1)
        amp = sim.statevector(np.zeros(2))
        np.testing.assert_allclose(np.abs(amp) ** 2,
                                   1.0 / self.feas.shape[0], atol=1e-12)

    def test_weight_is_conserved(self):
        sim = p3.WeightSubspaceSimulator(self.n, self.K, self.diag0, 3)
        probs = sim.prob_dict(self.rng.uniform(0, 6, size=6))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)
        for idx in probs:
            self.assertEqual(bin(int(idx)).count("1"), self.K)

    def test_energy_matches_qiskit_circuit(self):
        for p in (1, 2):
            theta = self.rng.uniform(0, 6, size=2 * p)
            sim = p3.WeightSubspaceSimulator(self.n, self.K, self.diag0, p)
            qc = p3.build_method_a_circuit(self.inst, self.cost0, p)
            full = Statevector(qaoa.bind_theta(qc, theta)).data
            sub = full[self.feas]
            norm = np.linalg.norm(sub)
            fid = abs(np.vdot(sub / norm, sim.statevector(theta)
                              / np.linalg.norm(sim.statevector(theta)))) ** 2
            self.assertAlmostEqual(fid, 1.0, places=12)
            self.assertLess(abs(1.0 - norm ** 2), 1e-10)
            e_full = float(np.sum(np.abs(sub) ** 2 * self.diag0[self.feas])
                           / norm ** 2)
            self.assertAlmostEqual(sim.energy(theta), e_full, places=10)


if __name__ == "__main__":
    unittest.main()
