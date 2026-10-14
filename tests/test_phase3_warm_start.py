"""Phase 3 tests: continuous relaxation, warm start, full-space simulator."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector

import phase1_core as core
import phase2_qaoa as qaoa
import phase3_methods as p3


class TestRelaxationAndWarmStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.bench = core.exact_benchmark(6, 3, cls.inst.mu, cls.inst.Sigma,
                                         cls.inst.q)
        cls.relax = p3.continuous_relaxation(cls.inst, 1.0)
        cls.rng = np.random.default_rng(55)

    def test_relaxation_bounds_and_residual(self):
        c = np.asarray(self.relax["final_point"])
        self.assertTrue(np.all(c >= -1e-9) and np.all(c <= 1 + 1e-9))
        self.assertLess(abs(self.relax["constraint_residual"]), 1.0)
        self.assertIsInstance(self.relax["success"], bool)

    def test_relaxation_is_a_lower_bound(self):
        # a relaxation minimises over a superset, so its value cannot exceed the
        # true feasible minimum (up to solver tolerance)
        self.assertLessEqual(self.relax["objective"],
                             self.bench["best_feasible_objective"] + 1e-6)

    def test_round_relaxation_is_feasible(self):
        r = p3.round_relaxation(self.inst, self.relax["final_point"])
        x = np.asarray(r["rounded_assignment"])
        self.assertEqual(int(x.sum()), self.inst.K)
        self.assertTrue(r["is_feasible"])
        self.assertAlmostEqual(r["objective"],
                              core.cost_unpenalized(x, self.inst.mu,
                                                    self.inst.Sigma, self.inst.q),
                              places=12)

    def test_product_state_marginals_and_norm(self):
        c = self.rng.uniform(0.05, 0.95, size=6)
        amp = p3.product_state_amplitudes(c)
        probs = np.abs(amp) ** 2
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)
        for i in range(6):
            marg = sum(probs[s] for s in range(64) if (s >> i) & 1)
            self.assertAlmostEqual(marg, c[i], places=12)

    def test_full_sim_matches_fast_sim_from_plus(self):
        cost = qaoa.cost_operator(self.inst, 1.0)
        diag = core.operator_diagonal(cost)
        for p in (1, 2, 3):
            theta = self.rng.uniform(0, 6, size=2 * p)
            a = p3.full_statevector(diag, 6, p, theta,
                                    p3.uniform_plus_amplitudes(6))
            b = qaoa.fast_statevector(6, p, diag, theta)
            self.assertAlmostEqual(abs(np.vdot(a, b)) ** 2, 1.0, places=12)

    def test_full_sim_product_state_matches_qiskit(self):
        cost = qaoa.cost_operator(self.inst, 1.0)
        diag = core.operator_diagonal(cost)
        c = self.rng.uniform(0.1, 0.9, size=6)
        init = p3.product_state_amplitudes(c)
        for p in (1, 2):
            theta = self.rng.uniform(0, 6, size=2 * p)
            qc = QuantumCircuit(6)
            for i in range(6):
                qc.ry(2 * np.arcsin(np.sqrt(c[i])), i)
            for k in range(p):
                qc.append(PauliEvolutionGate(cost, time=theta[k]), range(6))
                qc.append(PauliEvolutionGate(qaoa.build_x_mixer(6),
                                             time=theta[p + k]), range(6))
            sv = Statevector(qc).data
            vec = p3.full_statevector(diag, 6, p, theta, init)
            self.assertAlmostEqual(abs(np.vdot(sv, vec)) ** 2, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
