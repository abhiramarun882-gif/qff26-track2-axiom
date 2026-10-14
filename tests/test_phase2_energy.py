"""Phase 2 tests: exact energy evaluation and operator consistency."""

import unittest

import numpy as np

import phase1_core as core
import phase2_qaoa as qaoa


class TestEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.lam = 1.0
        cls.cost = qaoa.cost_operator(cls.inst, cls.lam)
        cls.p = 3
        cls.theta = [0.21, 0.73, 1.41, 0.34, 1.02, 2.11]

    def test_exact_energy_matches_independent_diagonal_contraction(self):
        vec = qaoa.qaoa_statevector(self.qc(), self.theta).data
        e_sv = qaoa.qaoa_energy(self.qc(), self.cost, self.theta)
        e_diag = qaoa.qaoa_energy_from_vector(self.cost, vec)
        self.assertAlmostEqual(e_sv, e_diag, places=10)

    def test_energy_is_deterministic(self):
        qc = self.qc()
        e1 = qaoa.qaoa_energy(qc, self.cost, self.theta)
        e2 = qaoa.qaoa_energy(qc, self.cost, self.theta)
        self.assertEqual(e1, e2)

    def test_energy_matches_qaoa_ansatz(self):
        self.assertAlmostEqual(
            qaoa.qaoa_energy(self.qc(), self.cost, self.theta),
            qaoa.qaoa_ansatz_energy(6, self.p, self.cost, self.theta),
            places=10)

    def test_operator_diagonal_equals_penalized_cost(self):
        diag = core.operator_diagonal(self.cost)
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            expected = core.cost_penalized(x, self.inst.mu, self.inst.Sigma,
                                           self.inst.K, self.lam, self.inst.q)
            self.assertAlmostEqual(float(diag[s]), expected, places=9)

    def test_uniform_superposition_energy_is_mean_diagonal(self):
        qc = qaoa.build_qaoa_circuit(6, 1, self.cost)
        e = qaoa.qaoa_energy(qc, self.cost, [0.0, 0.0])
        diag = core.operator_diagonal(self.cost)
        self.assertAlmostEqual(e, float(np.mean(diag)), places=10)

    def qc(self):
        return qaoa.build_qaoa_circuit(6, self.p, self.cost,
                                       qaoa.build_x_mixer(6))


class TestFastSimulator(unittest.TestCase):
    """The fast inner-loop simulator must match the Qiskit circuit exactly."""

    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.cost = qaoa.cost_operator(cls.inst, 1.0)
        cls.diag = core.operator_diagonal(cls.cost)
        cls.thetas = {
            1: [0.3, 0.8],
            2: [0.4, 1.1, 0.7, 1.9],
            3: [0.2, 0.9, 1.7, 0.5, 1.3, 2.4],
        }

    def test_fast_statevector_matches_qiskit(self):
        from qiskit.quantum_info import Statevector
        for p, theta in self.thetas.items():
            qc = qaoa.build_qaoa_circuit(6, p, self.cost, qaoa.build_x_mixer(6))
            sv_true = Statevector(qaoa.bind_theta(qc, theta)).data
            sv_fast = qaoa.fast_statevector(6, p, self.diag, theta)
            fid = abs(np.vdot(sv_true, sv_fast)) ** 2
            self.assertGreater(fid, 1 - 1e-10, msg=f"p={p}")

    def test_fast_energy_matches_qiskit(self):
        for p, theta in self.thetas.items():
            qc = qaoa.build_qaoa_circuit(6, p, self.cost, qaoa.build_x_mixer(6))
            sim = qaoa.FastQAOASimulator(6, p, self.diag)
            self.assertAlmostEqual(sim.energy(theta),
                                   qaoa.qaoa_energy(qc, self.cost, theta),
                                   places=10)

    def test_fast_probabilities_match_qiskit(self):
        p = 2
        theta = self.thetas[2]
        qc = qaoa.build_qaoa_circuit(6, p, self.cost, qaoa.build_x_mixer(6))
        sim = qaoa.FastQAOASimulator(6, p, self.diag)
        np.testing.assert_allclose(sim.probabilities(theta),
                                   qaoa.exact_probabilities(qc, theta),
                                   atol=1e-12)


if __name__ == "__main__":
    unittest.main()
