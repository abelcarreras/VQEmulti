from openfermion import MolecularData
from openfermionpyscf import run_pyscf, PyscfMolecularData
from vqemulti.utils import generate_reduced_hamiltonian
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.errors import NotConvergedError
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals, get_pauli_symmetry_reduced_pool
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
import unittest


class OperationsTest(unittest.TestCase):

    def setUp(self):
        h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                              ['H', [0, 0, 1.0]],
                                              ['H', [0, 0, 2.0]],
                                              ['H', [0, 0, 3.0]]],
                                    basis='sto-3g',
                                    multiplicity=1,
                                    charge=0,
                                    description='H4')

        # run classical calculation
        self.molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=False)
        print('molecule')

    def test_adaptvqe_exact_fermion(self):

        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = self.molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # get UCCSD pool
        operators_pool = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        # define ansatz
        adapt_ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

        # define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        from vqemulti.optimizers import OptimizerParams
        opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
        opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

        from vqemulti.method.adapt_vanila import AdapVanilla

        method = AdapVanilla(gradient_threshold=2e-1,
                             diff_threshold=0,
                             coeff_tolerance=1e-10,
                             operator_update_number=1,
                             operator_update_max_grad=2e-2,
                             min_iterations=2,
                             )

        print('Initialize adaptVQE bfgs')
        result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          adapt_ansatz,
                          energy_threshold=1e-2,
                          method=method,
                          max_iterations=8,
                          optimizer_params=opt_bfgs
                          )

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -2.09854594, places=8)
        self.assertAlmostEquals(result['energy'], -2.15794641, places=4)
        self.assertAlmostEquals(self.molecule.fci_energy,  -2.16638745, places=8)

        self.assertEqual(len(result['ansatz']), 4)

        print('Initialize adaptVQE cobyla')
        result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          adapt_ansatz,
                          energy_threshold=1e-2,
                          method=method,
                          max_iterations=8,
                          optimizer_params=opt_cobyla
                          )

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -2.09854594, places=8)
        self.assertAlmostEquals(result['energy'], -2.15794641, places=4)
        self.assertAlmostEquals(self.molecule.fci_energy, -2.16638745, places=8)

        self.assertEqual(len(result['ansatz']), 4)

    def test_adaptvqe_exact_qubit_symmetry(self):
        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = self.molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # define fermionic pool: singlet-adapted singlet and double excitations
        operators_pool = get_pool_singlet_sd(n_electrons, n_orbitals)
        print('operators_pool: ', len(operators_pool))
        self.assertEqual(len(operators_pool), 14)

        # get reduced fermionic pool by point symmetry
        sym_orbitals = symmetrize_molecular_orbitals(self.molecule, 'D2h', skip=True)
        operators_pool = get_symmetry_reduced_pool(operators_pool, sym_orbitals, threshold=0.5)  # using symmetry
        print('operators_pool_sym: ', len(operators_pool))
        self.assertEqual(len(operators_pool), 8)

        # get qubit pool
        operators_pool = operators_pool.get_quibits_list(normalize=True) # get qubits
        print('operators_pool_qubit: ', len(operators_pool))
        self.assertEqual(len(operators_pool), 88)

        # get reduced qubit pool by pauli symmetry
        operators_pool = get_pauli_symmetry_reduced_pool(operators_pool)
        print('operators_pool_qubit_sym: ', len(operators_pool))
        self.assertEqual(len(operators_pool), 44)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        # define ansatz
        adapt_ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

        # define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        from vqemulti.optimizers import OptimizerParams
        opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
        opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

        from vqemulti.method.adapt_vanila import AdapVanilla

        method = AdapVanilla(gradient_threshold=2e-1,
                             diff_threshold=0,
                             coeff_tolerance=1e-10,
                             operator_update_number=1,
                             operator_update_max_grad=2e-2,
                             min_iterations=2,
                             )

        print('Initialize adaptVQE bfgs')
        try:
            result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                              operators_pool,  # fermionic operators
                              adapt_ansatz,
                              energy_threshold=1e-2,
                              method=method,
                              max_iterations=8,
                              optimizer_params=opt_bfgs
                              )

        except NotConvergedError as c:
            result = c.results

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -2.09854594, places=8)
        self.assertAlmostEquals(result['energy'], -2.16372766, places=4)
        self.assertAlmostEquals(self.molecule.fci_energy, -2.16638745, places=8)

        self.assertEqual(len(result['ansatz']), 8)

    def test_adaptvqe_exact_fermion_simulation(self):

        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = self.molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # get UCCSD ansatz
        operators_pool = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        # define ansatz
        adapt_ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

        # define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        from vqemulti.optimizers import OptimizerParams
        opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
        # opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

        # setup simulator
        simulator = QiskitSimulator(trotter=False, trotter_steps=1, test_only=True)

        from vqemulti.method.adapt_vanila import AdapVanilla
        method = AdapVanilla(gradient_threshold=2e-1,
                             diff_threshold=0,
                             coeff_tolerance=1e-10,
                             operator_update_number=1,
                             operator_update_max_grad=2e-2,
                             min_iterations=2,
                             gradient_simulator=simulator,
                             )

        print('Initialize adaptVQE bfgs')
        result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          adapt_ansatz,
                          energy_threshold=1e-2,
                          method=method,
                          max_iterations=8,
                          optimizer_params=opt_bfgs,
                          energy_simulator=simulator,
                          )

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -2.09854594, places=8)
        self.assertAlmostEquals(result['energy'], -2.15794641, places=4)
        self.assertAlmostEquals(self.molecule.fci_energy,  -2.16638745, places=8)

        self.assertEqual(len(result['ansatz']), 4)

    def test_adaptvqe_simulation_trotter(self):
        # get properties from classical SCF calculation
        n_electrons = self.molecule.n_electrons
        n_orbitals = self.molecule.n_orbitals

        hamiltonian = self.molecule.get_molecular_hamiltonian()

        print('n_electrons: ', n_electrons)
        print('n_orbitals: ', n_orbitals)
        print('n_qubits:', hamiltonian.n_qubits)

        # get UCCSD ansatz
        operators_pool = get_pool_singlet_sd(n_electrons, n_orbitals)

        # Get reference Hartree Fock state
        hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

        # define ansatz
        adapt_ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

        # define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        from vqemulti.optimizers import OptimizerParams
        opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
        # opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

        # setup simulator
        simulator = QiskitSimulator(trotter=True, trotter_steps=1, test_only=True)

        from vqemulti.method.adapt_vanila import AdapVanilla

        method = AdapVanilla(gradient_threshold=2e-1,
                             diff_threshold=0,
                             coeff_tolerance=1e-10,
                             operator_update_number=1,
                             operator_update_max_grad=2e-2,
                             min_iterations=2,
                             gradient_simulator=simulator,
                             )

        print('Initialize adaptVQE bfgs')
        result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          adapt_ansatz,
                          energy_threshold=1e-2,
                          method=method,
                          max_iterations=8,
                          optimizer_params=opt_bfgs,
                          energy_simulator=simulator,
                          )

        print('Energy HF: {:.8f}'.format(self.molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy FullCI: {:.8f}'.format(self.molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Operators:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])

        self.assertAlmostEquals(self.molecule.hf_energy, -2.09854594, places=8)
        self.assertAlmostEquals(result['energy'], -2.15794641, places=4)
        self.assertAlmostEquals(self.molecule.fci_energy,  -2.16638745, places=8)

        self.assertEqual(len(result['ansatz']), 4)

