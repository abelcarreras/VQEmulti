from vqemulti.ansatz.generators.basis import get_spin_matrix, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
from vqemulti.ansatz.generators import get_ucc_generator
from vqemulti.ansatz import GenericAnsatz
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.preferences import Configuration
from vqemulti.utils import log_section, print_tensor_4d
from vqemulti.ansatz.unitary_jastrow import get_basis_change_exp
from scipy.linalg import expm
import numpy as np
import scipy as sp


class HardwareEfficientAnsatz(GenericAnsatz):
    """
    ansatz type: (e^k e^iJ) * n_terms

    """
    def __init__(self, hf_reference_fock,
                 init='zero',
                 n_terms=None,
                 mixed_spin=True,
                 complex_rotation=False,
                 local=None,
                 connectivity_graph=None,
                 separate_spins=False,
                 ):
        """
        assumed HF as reference

        :param hf_reference_fock: reference vector in fock space
        :param full_trotter: trotterize exponent (necessary for circuit implmentation)
        :param use_qubit: transform fermion to qubit operators early (deprecated)
        :param n_terms: number of UCC layers used
        :param local: do a local version of the J operators (0:all zeros, 1: diagonal, 2: tridigonal, etc...)
        :param separate_spins: separate spin operators approach (under testing: incorrect phases)
        :param mixed_spin: include mixed spin interactions
        :param complex_rotation: include complex valued rotations (uses more paramters)
        """
        super().__init__()
        self._operators = []
        self._parameters = []
        self._matrices = []
        self._spin_t1 = None

        self._reference_fock = hf_reference_fock
        self._mixed_spin = mixed_spin
        self._n_terms = n_terms
        self._complex_rotation = complex_rotation
        self._separate_spins = separate_spins

        n_orb = len(hf_reference_fock)//2

        n_param_k, n_param_j = self.get_param_size()
        # print('n_param_k:', n_param_k)
        # print('n_param_j:', n_param_j)

        # initialize parameters
        # n_param = n_param_k * ( 1 + (self._n_terms-1)) + n_param_j * (self._n_terms-1)
        n_param = (n_param_k + n_param_j) * self._n_terms

        if init=='zeros':
            self._parameters = [0.0] * n_param
        elif init=='ones':
            self._parameters = [1.0] * n_param
        elif init=='random':
            self._parameters = np.random.rand(n_param)
        else:
            raise ValueError('init must be either zeros, ones, or random')

        self._operators, self._matrices = self._get_matrices(self._parameters)
        self._mask = [True] * n_param

        def get_mask(n_orb, G, local):

            """
            Return a boolean mask for the parameter vector corresponding to
            np.triu_indices(size).
            """
            import networkx as nx

            mask_total = np.ones(n_param, dtype=bool)

            for i_term in range(n_terms):

                mask = np.zeros(n_orb * (n_orb + 1) // 2, dtype=bool)
                i, j = np.triu_indices(n_orb)
                if G is None:
                    # Distance along the diagonal
                    mask = (j - i) < local
                else:
                    distances = dict(nx.all_pairs_shortest_path_length(G))
                    for k, (ii, jj) in enumerate(zip(i, j)):
                        if distances[ii][jj] <= local - 1:
                            mask[k] = True

                n = (n_param_k + n_param_j) * i_term + n_param_k
                mask_total[n: n+n_param_j] = mask
            return mask_total

        # set mask for local
        if local is not None:
            self._mask = get_mask(n_orb, connectivity_graph, local)
            for i, t in enumerate(self._mask):
                if not t: self._parameters[i] = 0.0

    def get_param_size(self):
        n_orb = len(self._reference_fock)//2

        if self._complex_rotation:
            n_param_k = n_orb**2
        else:
            n_param_k = (n_orb**2 - n_orb)//2
        n_param_j = (n_orb**2 - n_orb)//2 + n_orb

        return n_param_k, n_param_j

    @property
    def n_qubits(self):
        return len(self._reference_fock)

    @property
    def operators(self):
        return self._operators

    @property
    def parameters(self):
        assert len(self._parameters) == len(self._mask)
        return np.asarray(self._parameters)[self._mask]

    @parameters.setter
    def parameters(self, parameters):
        assert len(parameters) == sum(self._mask)
        assert len(self._parameters) == len(self._mask)
        params = np.asarray(self._parameters)
        params[self._mask] = parameters
        self._parameters = params.tolist()

        self._operators, self._matrices = self._get_matrices(self._parameters)

    def set_mask(self, mask):
        self._mask = mask

    def add_term(self, init):
        """
        add term to the ansatz

        :param init: zeros, ones or random
        """

        param = self._parameters

        self.__init__(self._reference_fock, init, n_terms=self._n_terms+1, mixed_spin=self._mixed_spin, complex_rotation=self._complex_rotation)
        new_param = self.parameters
        new_param[:len(param)] = param
        self.parameters = new_param


    def _get_matrices(self, parameters):

        operators = []

        # bind parameters
        n_param_k, n_param_j = self.get_param_size()
        n_orb = self.n_qubits // 2

        def generator_from_parameters_real(parameters, size):
            kappa = np.zeros((size, size))
            i, j = np.triu_indices(size, k=1)

            kappa[i, j] = parameters
            kappa[j, i] = [-p for p in parameters]

            return kappa

        def generator_from_parameters_complex(parameters, size):
            # Number of independent pairs
            n_pairs = size * (size - 1) // 2
            assert len(parameters) == 2 * n_pairs + size

            real = np.asarray(parameters[:n_pairs], dtype=float)
            imag = np.asarray(parameters[n_pairs:n_pairs * 2], dtype=float)
            imag_diag = np.asarray(parameters[n_pairs * 2:], dtype=float)

            kappa = np.zeros((size, size), dtype=complex)
            i, j = np.triu_indices(size, k=1)

            # Upper triangle
            kappa[i, j] = real + 1j * imag

            # Lower triangle (anti-Hermitian)
            kappa[j, i] = -real + 1j * imag

            i, j = np.diag_indices(size)
            kappa[i, j] = 1j * imag_diag

            return kappa

        def generator_from_parameters(parameters, size):
            if self._complex_rotation:
                return generator_from_parameters_complex(parameters, size)
            else:
                return generator_from_parameters_real(parameters, size)

        def unitary_from_parameters(parameters, size):
            kappa = generator_from_parameters(parameters, size)
            return expm(kappa)


        def symmetric_from_parameters(parameters, size):

            mat = np.zeros((size, size))
            i, j = np.triu_indices(size)

            mat[i, j] = parameters
            mat[j, i] = parameters

            from scipy.linalg import expm
            return mat


        # basis change
        pos = 0
        matrices = []

        for i in range(self._n_terms):

            # basis change
            # old version
            # U_i = unitary_from_parameters(parameters[pos:n_param_k+pos], n_orb)
            # U_spin = get_spin_matrix(U_i)
            # ansatz_u = get_basis_change_exp(U_spin, use_qubit=False)  # a_i^ a_j

            kappa_i = generator_from_parameters(parameters[pos:n_param_k+pos], n_orb)
            kappa_spin = get_spin_matrix(kappa_i)
            ansatz_u = get_ucc_generator(-kappa_spin, None, full_amplitudes=True, tolerance=1e-6)
            U_spin = expm(kappa_spin)

            matrices.append(('K', U_spin))
            operators.append(ansatz_u)
            pos += n_param_k

            # jastrow
            diag_i = symmetric_from_parameters(parameters[pos:n_param_j+pos], n_orb)

            j_mat = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
            for i in range(n_orb):
                for j in range(n_orb):
                    j_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

            spin_jastrow = get_t2_spinorbitals_absolute_full(j_mat, mixed_spin=self._mixed_spin)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
            ansatz_j = get_ucc_generator(None, spin_jastrow, full_amplitudes=True)

            matrices.append(('J', ansatz_j))
            operators.append(ansatz_j)
            pos += n_param_j

        return operators, matrices


    def get_preparation_gates(self, simulator):

        if Configuration().mapping == 'jw':
            # this is only for JW mapping (due to givens rotations implementation)

            state_preparation_gates = simulator.get_reference_gates(self._reference_fock)


            #self._operators, self._matrices = self._get_matrices(self._parameters)

            for matrix, operator in zip(self._matrices, self._operators):

                if matrix[0] == 'K':
                    # implement rotation term
                    rotation_p = matrix[1].T.conj()
                    state_preparation_gates += simulator.get_rotation_gates(rotation_p,
                                                                            self.n_qubits,
                                                                            separate_spins=self._separate_spins)

                elif matrix[0] == 'J':
                    # implement jastrow term
                    jastrow_qubit = operator.get_quibits_list(reorganize=False)
                    state_preparation_gates += simulator.get_exponential_gates(jastrow_qubit, self.n_qubits)
                else:
                    raise NotImplementedError

            return state_preparation_gates

        else:
            return super().get_preparation_gates(simulator)

    def _simulate_energy(self, hamiltonian, simulator, return_std=False):
        """
        Obtain the hamiltonian expectation value for a given VQE state (reference + ansatz) and a hamiltonian

        :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
        :param simulator: simulation object
        :param return_std: return std also
        :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
        """
        from vqemulti.utils import fermion_to_qubit

        # transform to qubit hamiltonian
        qubit_hamiltonian = fermion_to_qubit(hamiltonian)

        # get gates to prepare the state
        state_preparation_gates = self.get_preparation_gates(simulator)

        # evaluate hamiltonian
        energy, std_error = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        if return_std:
            return energy, std_error

        return energy


    def get_state_vector(self):
        """
        prepare state vector from coefficients ansatz and reference

        :return state vector
        """
        from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator

        # Initialize the state vector with the reference state.
        # |state> = |state_old> · exp(i * coef · |ansatz>
        # imaginary i already included in |ansatz>
        state = get_sparse_ket_from_fock(self._reference_fock)

        # Apply the ansatz operators one by one to obtain the state as optimized by the last iteration
        for operator in self._operators:
            sparse_operator = get_sparse_operator(operator[0], self.n_qubits)
            state = sp.sparse.linalg.expm_multiply(sparse_operator, state)
        return state

    def get_sampling(self, simulator):

        state_preparation_gates = self.get_preparation_gates(simulator)
        sampling = simulator.get_state_sampling(state_preparation_gates, self.n_qubits)

        return sampling



if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer import AerSimulator

    config = Configuration()
    #config.verbose = 2
    config.mapping = 'jw'

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True, shots=10000,
                          # backend=FakeTorino(),
                          # use_ibm_runtime=True
                          )

    simulator_sqd = simulator.copy()
    simulator_sqd._backend = AerSimulator()
    simulator_sqd._use_ibm_runtime = True

    hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [2.0, 0.0, 0.0]),
                                       ('H', [4.0, 0.0, 0.0]),
                                       ('H', [6.0, 0.0, 0.0])],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='molecule')

    # run classical calculation
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=0, n_orbitals=4, run_ccsd=True)

    n_electrons = molecule.n_electrons
    hamiltonian = molecule.get_molecular_hamiltonian()



    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, molecule.n_qubits)
    hea = HardwareEfficientAnsatz(hf_reference_fock, init='zero', n_terms=1, mixed_spin=True)

    print(hea.parameters)
    #param = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    #hea.parameters = param

    # simulator = None
    energy = hea.get_energy(hea.parameters, hamiltonian, simulator)


    print('HEA energy: ', energy)
    #exit()

    simulator.print_statistics()
    print(simulator.get_circuits()[-1])

