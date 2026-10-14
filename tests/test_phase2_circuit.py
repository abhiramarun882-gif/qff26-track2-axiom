"""Phase 2 tests: explicit QAOA circuit construction and conventions."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz
from qiskit.quantum_info import Statevector

import phase1_core as core
import phase2_qaoa as qaoa


class TestQAOACircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inst = core.generate_instance(6, 3)
        cls.lam = 1.0
        cls.cost = qaoa.cost_operator(cls.inst, cls.lam)

    def test_qubit_count(self):
        qc = qaoa.build_qaoa_circuit(6, 2, self.cost)
        self.assertEqual(qc.circuit.num_qubits, 6)

    def test_parameter_count(self):
        for p in (1, 2, 3, 4):
            qc = qaoa.build_qaoa_circuit(6, p, self.cost)
            self.assertEqual(qc.num_parameters, 2 * p)
            self.assertEqual(len(qc.circuit.parameters), 2 * p)

    def test_parameter_vector_order_is_gamma_then_beta(self):
        # ``QuantumCircuit.parameters`` iteration order is not guaranteed to be
        # insertion order, so binding is always done with an explicit dict. The
        # documented flat-vector convention is gammas first, then betas.
        qc = qaoa.build_qaoa_circuit(6, 3, self.cost)
        self.assertEqual(qc.theta_order, list(qc.gamma) + list(qc.beta))
        names = [p.name for p in qc.theta_order]
        self.assertEqual(names, [f"gamma[{k}]" for k in range(3)]
                         + [f"beta[{k}]" for k in range(3)])

    def test_initial_state_is_plus(self):
        qc = qaoa.build_qaoa_circuit(6, 1, self.cost)
        sv = qaoa.qaoa_statevector(qc, [0.0, 0.0])
        expected = Statevector.from_label("+" * 6)
        self.assertGreater(abs(sv.inner(expected)), 1 - 1e-12)

    def test_mixer_is_transverse_field_x(self):
        # gamma = 0 -> only the mixer acts; compare to rx(2 beta) per qubit.
        p, b = 1, 0.37
        qc = qaoa.build_qaoa_circuit(6, p, self.cost)
        sv = qaoa.qaoa_statevector(qc, [0.0, b])
        ref = QuantumCircuit(6)
        ref.h(range(6))
        for i in range(6):
            ref.rx(2.0 * b, i)
        self.assertGreater(abs(sv.inner(Statevector(ref))), 1 - 1e-12)

    def test_mixer_operator_is_sum_x(self):
        mixer = qaoa.build_x_mixer(6)
        self.assertEqual(mixer.num_qubits, 6)
        for label in mixer.paulis.to_labels():
            self.assertEqual(label.count("X"), 1)
        self.assertEqual(mixer.size, 6)

    def test_layer_order_matches_qaoa_ansatz(self):
        p = 2
        theta = [0.11, 0.22, 0.33, 0.44]
        qc = qaoa.build_qaoa_circuit(6, p, self.cost)
        sv1 = qaoa.qaoa_statevector(qc, theta)

        init = QuantumCircuit(6)
        init.h(range(6))
        ansatz = QAOAAnsatz(cost_operator=self.cost, reps=p,
                            initial_state=init,
                            mixer_operator=qaoa.build_x_mixer(6))
        # QAOAAnsatz order is [beta..., gamma...]
        bound = ansatz.assign_parameters([theta[2], theta[3], theta[0], theta[1]])
        self.assertGreater(abs(sv1.inner(Statevector(bound))), 1 - 1e-10)

        self.assertAlmostEqual(
            qaoa.qaoa_energy(qc, self.cost, theta),
            qaoa.qaoa_ansatz_energy(6, p, self.cost, theta), places=10)

    def test_parameter_binding_correspondence(self):
        p = 2
        theta = [0.1, 0.2, 0.3, 0.4]
        qc = qaoa.build_qaoa_circuit(6, p, self.cost)
        gammas, betas = qc.split_theta(theta)
        np.testing.assert_allclose(gammas, [0.1, 0.2])
        np.testing.assert_allclose(betas, [0.3, 0.4])

        manual = QuantumCircuit(6)
        manual.h(range(6))
        for k in range(p):
            manual.append(PauliEvolutionGate(self.cost, time=gammas[k]), range(6))
            for i in range(6):
                manual.rx(2.0 * betas[k], i)
        sv = qaoa.qaoa_statevector(qc, theta)
        self.assertGreater(abs(sv.inner(Statevector(manual))), 1 - 1e-12)
        self.assertEqual(len(qaoa.bind_theta(qc, theta).parameters), 0)

    def test_bitstring_decoding_matches_phase1(self):
        for s in range(2 ** 6):
            x = core.index_to_assets(s, 6)
            self.assertEqual(core.assets_to_index(x), s)


if __name__ == "__main__":
    unittest.main()
