from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.energy import exact_vqe_energy
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from vqemulti.simulators.penny_simulator import PennylaneSimulator
from vqemulti.simulators.cirq_simulator import CirqSimulator
from vqemulti.energy import simulate_vqe_energy
from vqemulti import vqe
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
        molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

        # get properties from classical SCF calculation
        n_electrons = molecule.n_electrons
        n_orbitals = 3  # molecule.n_orbitals

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)

        self.hamiltonian = molecule.get_molecular_hamiltonian()
        self.hamiltonian = generate_reduced_hamiltonian(self.hamiltonian, n_orbitals, frozen_core=0)

        print('n_qubits:', self.hamiltonian.n_qubits)

        # Get UCCSD ansatz
        self.uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=0)

        # Get reference Hartree Fock state
        print(n_electrons, self.hamiltonian.n_qubits)

        self.hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, self.hamiltonian.n_qubits, frozen_core=0)
        print('hf reference', self.hf_reference_fock)

        # Compute VQE to get a wave function (coefficients/ansatz)  [no simulation]
        print('Initialize VQE')

        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            self.result = vqe(self.hamiltonian, self.uccsd_ansatz[:3], self.hf_reference_fock, energy_simulator=simulator)

            print('Simulator: ', simulator)
            print('Coefficients: ', self.result['coefficients'])

            print('Energy HF: {:.8f}'.format(molecule.hf_energy))
            print('Energy VQE: {:.8f}'.format(self.result['energy']))
            print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
            print('------------------------')

            self.assertAlmostEquals(molecule.hf_energy, -1.09138607, places=4)
            self.assertAlmostEquals(self.result['energy'], -1.10717770, places=4)
            self.assertAlmostEquals(molecule.fci_energy, -1.123253503, places=4)

    def test_vqe_optimal(self):

        print('test HF reference optimal')
        energy_sp = exact_vqe_energy([], self.uccsd_ansatz[:0],
                                     self.hf_reference_fock,
                                     self.hamiltonian)

        self.assertAlmostEquals(energy_sp, -1.09138607, places=4)
        print('SP Energy: {:.8f}'.format(energy_sp))

        print('test 3 operators ansatz optimal')
        energy_sp = exact_vqe_energy(self.result['coefficients'],
                                     self.result['ansatz'],
                                     self.hf_reference_fock,
                                     self.hamiltonian)
        print('SP Energy: {:.8f}'.format(energy_sp))
        self.assertAlmostEquals(energy_sp, -1.10717597, places=4)

    def test_vqe_simulators_exact(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=False, test_only=True, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy([], self.uccsd_ansatz[:0],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.09138607, places=4)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        simulator = QiskitSimulator(trotter=False, test_only=True, hamiltonian_grouping=True)

        energy_sp = simulate_vqe_energy(self.result['coefficients'],
                                        self.result['ansatz'],
                                        self.hf_reference_fock,
                                        self.hamiltonian, simulator)

        print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
        self.assertAlmostEquals(energy_sp, -1.10717927, places=4)

    def test_vqe_simulators_exact_trotter(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy([], self.uccsd_ansatz[:0],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.09138607, places=4)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy(self.result['coefficients'],
                                            self.result['ansatz'],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.10717931, places=4)

    def test_vqe_simulators_sampling(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=False, test_only=False, shots=100000, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy([], self.uccsd_ansatz[:0],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.09138607, places=2)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=False, test_only=False, shots=100000, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy(self.result['coefficients'],
                                            self.result['ansatz'],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.10717931, places=2)

    def test_vqe_simulators_sampling_trotter(self):
        # HF reference solution

        print('test HF reference exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=False, shots=100000, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy([], self.uccsd_ansatz[:0],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.09138607, places=2)

        print('test 3 operators ansatz exact circuit evaluation')
        # simulators exact energy
        for Simulator in [QiskitSimulator, PennylaneSimulator, CirqSimulator]:
            simulator = Simulator(trotter=True, test_only=False, shots=100000, hamiltonian_grouping=True)

            energy_sp = simulate_vqe_energy(self.result['coefficients'],
                                            self.result['ansatz'],
                                            self.hf_reference_fock,
                                            self.hamiltonian, simulator)

            print('SP Energy {}: {:.8f}'.format(Simulator, energy_sp))
            self.assertAlmostEquals(energy_sp, -1.10717931, places=2)
