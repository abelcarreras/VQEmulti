from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from vqemulti.simulators.penny_simulator import PennylaneSimulator
from vqemulti.simulators.cirq_simulator import CirqSimulator
from vqemulti import vqe
import numpy as np
import unittest


class OperationsTest(unittest.TestCase):

    def setUp(self):

        # define molecule
        h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                              ['H', [0, 0, 1.0]]],
                                    basis='3-21g',
                                    multiplicity=1,
                                    charge=0,
                                    description='H2'
                                    )

        # run classical calculation
        molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True, n_orbitals=3)

        # get properties from classical SCF calculation
        n_electrons = molecule.n_electrons
        n_orbitals = molecule.n_orbitals
        n_qubits = molecule.n_qubits

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', n_qubits)

        self.hamiltonian = molecule.get_molecular_hamiltonian()

        # Get UCCSD ansatz
        from vqemulti.ansatz.generators import get_ucc_generator
        from vqemulti.ansatz.exponential import ExponentialAnsatz
        from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
        uccsd_pool = get_pool_singlet_sd(n_electrons, n_orbitals)

        coefficients = np.zeros_like(uccsd_pool)
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, self.hamiltonian.n_qubits)
        self.uccsd_ansatz = ProductExponentialAnsatz(coefficients, uccsd_pool, hf_reference_fock)
        print('coefficients: ', coefficients)

        # Get reference Hartree Fock state
        print('n_electrons', n_electrons)
        print('n_qubits', self.hamiltonian.n_qubits)

        # Compute VQE to get a wave function (coefficients/ansatz)  [no simulation]
        print('Initialize VQE')

        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            self.result = vqe(self.hamiltonian, self.uccsd_ansatz[:3], energy_simulator=simulator)

            print('Simulator: ', simulator)
            print('Coefficients: ', self.result['coefficients'])

            print('Energy HF: {:.8f}'.format(molecule.hf_energy))
            print('Energy VQE: {:.8f}'.format(self.result['energy']))
            print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
            print('------------------------')

            self.assertAlmostEqual(molecule.hf_energy, -1.09138607, places=4)
            self.assertAlmostEqual(self.result['energy'], -1.10717770, places=4)
            self.assertAlmostEqual(molecule.fci_energy, -1.123253503, places=4)

    def test_vqe_optimal(self):

        print('test HF reference optimal')
        energy_sp = self.uccsd_ansatz[:0].get_energy([], self.hamiltonian, None)
        print('SP Energy: {:.8f}'.format(energy_sp))
        self.assertAlmostEqual(energy_sp, -1.09138607, places=4)
        print('SP Energy: {:.8f}'.format(energy_sp))

        print('test 3 operators ansatz optimal')
        energy_sp = self.uccsd_ansatz[:3].get_energy(self.result['coefficients'], self.hamiltonian, None)

        print('SP Energy: {:.8f}'.format(energy_sp))
        self.assertAlmostEqual(energy_sp, -1.10717597, places=4)

    def test_vqe_simulators_exact(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator]:
            simulator = Simulator(trotter=False, test_only=True, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:0].get_energy([], self.hamiltonian, simulator)
            print('ene: ', energy_sp)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.09138607, places=4)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        simulator = QiskitSimulator(trotter=False, test_only=True, hamiltonian_grouping=True)

        print(self.uccsd_ansatz.parameters, len(self.uccsd_ansatz.operators))
        energy_sp = self.uccsd_ansatz[:3].get_energy(self.result['coefficients'], self.hamiltonian, simulator)

        print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
        self.assertAlmostEqual(energy_sp, -1.10717927, places=4)

    def test_vqe_simulators_exact_trotter(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:0].get_energy([], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.09138607, places=4)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:3].get_energy(self.result['coefficients'], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.10717931, places=4)

    def test_vqe_simulators_sampling(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=False, test_only=False, shots=1000000, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:0].get_energy([], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.09138607, places=2)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=False, test_only=False, shots=1000000, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:3].get_energy(self.result['coefficients'], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.10717931, places=2)

    def test_vqe_simulators_sampling_trotter(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=False, shots=1000000, hamiltonian_grouping=True)

            energy_sp = self.uccsd_ansatz[:0].get_energy([], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.09138607, places=2)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=False, shots=1000000, hamiltonian_grouping=True)
            energy_sp = self.uccsd_ansatz[:3].get_energy(self.result['coefficients'], self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEqual(energy_sp, -1.10717931, places=2)
