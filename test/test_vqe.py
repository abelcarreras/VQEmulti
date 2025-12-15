from openfermion import MolecularData
from openfermionpyscf import run_pyscf, PyscfMolecularData
from vqemulti.utils import generate_reduced_hamiltonian
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.utils import get_hf_reference_in_fock_space, fermion_to_qubit
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from vqemulti.preferences import Configuration
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

    def test_vqe_exact_qubit_jw(self):

        Configuration().mapping = 'jw' # Jordan-Wigner mapping
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
        uccsd_ansatz = uccsd_ansatz.get_quibits_list() # to qubit operators

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
        hamiltonian = fermion_to_qubit(hamiltonian) # to qubit operators

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock)

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

        self.assertEqual(len(result['ansatz']), 12)

    def test_vqe_exact_qubit_bk(self):

        Configuration().mapping = 'bk'  # Bravyi-Kitaev mapping
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
        uccsd_ansatz = uccsd_ansatz.get_quibits_list() # to qubit operators

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
        hamiltonian = fermion_to_qubit(hamiltonian) # to qubit operators

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock)

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

        self.assertEqual(len(result['ansatz']), 12)

    def test_vqe_exact_qubit_parity(self):

        Configuration().mapping = 'pc'  # parity mapping
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
        uccsd_ansatz = uccsd_ansatz.get_quibits_list()  # to qubit operators

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
        hamiltonian = fermion_to_qubit(hamiltonian)  # to qubit operators

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock)

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

        self.assertEqual(len(result['ansatz']), 12)

    def test_vqe_qubit_trotter(self):

        Configuration().mapping = 'jw'
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
        uccsd_ansatz = uccsd_ansatz.get_quibits_list() # to qubit operators

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
        hamiltonian = fermion_to_qubit(hamiltonian) # to qubit operators

        # setup simulator
        simulator = QiskitSimulator(trotter=True, trotter_steps=1, test_only=True)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
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

        Configuration().mapping = 'jw'

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
                     hf_reference_fock)

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(result['energy'], -1.12988981, places=6)
        self.assertEqual(len(result['ansatz']), 2)

    def test_vqe_trotter_fermion(self):

        Configuration().mapping = 'jw'

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

        # setup simulator
        simulator = QiskitSimulator(trotter=True, trotter_steps=1, test_only=True)

        print('Initialize VQE')
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     energy_simulator=simulator)

        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(result['energy'], -1.13557013, places=6)

        self.assertEqual(len(result['ansatz']), 5)
