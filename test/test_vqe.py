from openfermion import MolecularData
from openfermionpyscf import run_pyscf, PyscfMolecularData
from vqemulti.utils import generate_reduced_hamiltonian
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti import vqe
import unittest


class OperationsTest(unittest.TestCase):

    def setUp(self):
        h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                              ['H', [0, 0, 0.74]]],
                                    basis='3-21g',
                                    # basis='sto-3g',
                                    multiplicity=1,
                                    charge=0,
                                    description='H2')

        # run classical calculation
        self.molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    def test_vqe_exact_qubit(self):

        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = 2 # molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()
        hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # define UCCSD ansatz
        uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     opt_qubits=False)

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy CCSD: {:.8f}'.format(self.molecule.ccsd_energy))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -1.12294026, places=8)
        self.assertAlmostEquals(result['energy'], -1.12988983, places=4)
        self.assertAlmostEquals(self.molecule.ccsd_energy, -1.14781330, places=8)
        self.assertAlmostEquals(self.molecule.fci_energy, -1.14781313, places=8)

        self.assertEqual(len(result['ansatz']), 2)

    def test_vqe_qubit_trotter(self):
        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = 2  # molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()
        hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # define UCCSD ansatz
        uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        from vqemulti.simulators.penny_simulator import PennylaneSimulator
        simulator = PennylaneSimulator(trotter=True, trotter_steps=1, test_only=True)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     opt_qubits=True,
                     energy_simulator=simulator)

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy CCSD: {:.8f}'.format(self.molecule.ccsd_energy))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -1.12294026, places=8)
        self.assertAlmostEquals(result['energy'], -1.12988977, places=6)
        self.assertAlmostEquals(self.molecule.ccsd_energy, -1.14781330, places=8)
        self.assertAlmostEquals(self.molecule.fci_energy, -1.14781313, places=8)

        self.assertEqual(len(result['ansatz']), 12)

    def test_vqe_exact_fermion(self):
        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = 2  # molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()
        hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # define UCCSD ansatz
        uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     opt_qubits=False)

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(result['energy'], -1.12988981, places=6)
        self.assertEqual(len(result['ansatz']), 2)

    def test_vqe_trotter_fermion(self):
        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = 3  # self.molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()
        hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # define UCCSD ansatz
        uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        from vqemulti.simulators.penny_simulator import PennylaneSimulator
        simulator = PennylaneSimulator(trotter=True, trotter_steps=1, test_only=True)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     opt_qubits=False,
                     energy_simulator=simulator)

        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(result['energy'], -1.13557013, places=6)

        self.assertEqual(len(result['ansatz']), 5)
