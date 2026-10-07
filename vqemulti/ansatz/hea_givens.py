from vqemulti.ansatz.generators.basis import get_spin_matrix, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
from vqemulti.ansatz.generators import get_ucc_generator
from vqemulti.ansatz.hea_particle import HardwareEfficientAnsatz
from vqemulti.ansatz import GenericAnsatz

from vqemulti.simulators.qiskit_functions import QCTRLSampler
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.preferences import Configuration
from vqemulti.utils import log_section, print_tensor_4d
from vqemulti.ansatz.unitary_jastrow import get_basis_change_exp
from scipy.linalg import expm
import numpy as np
import scipy as sp
from openfermion.linalg import givens_decomposition_square


def set_givens(G, i, j, theta, phi):
    c = np.cos(theta)
    s = np.sin(theta)
    p = np.exp(1j * phi)

    G[i, i] = c  # cos θ
    G[i, j] = -p * s  # -e^{iφ} sin θ
    G[j, i] = s  # sin θ  (no phase!)
    G[j, j] = p * c  # e^{iφ} cos θ  (phase here, not above!)


def from_givens_to_U(givens_rotations, diagonal, n):

    U_work = np.eye(n, dtype=complex)
    for layer in givens_rotations[::-1]:
        for (i, j, theta, phi) in layer:
            G = np.eye(n, dtype=complex)
            ii = n-i-1
            jj = n-j-1
            set_givens(G, ii, jj, theta, phi)
            U_work = G @ U_work

    D = np.diag(diagonal)
    return D @ U_work


def get_givens_pack(params, givens_rotations, n_orb, real_rotations=False):

    givens_rotations = givens_rotations.copy()

    k = 0
    for i, layer in enumerate(givens_rotations):

        rotation = []
        for gr in layer:
            if real_rotations:
                rotation.append((gr[0], gr[1], params[k], 0.0))
                k += 1
            else:
                rotation.append((gr[0], gr[1], params[k], params[k+1]))
                k += 2

        givens_rotations[i] = tuple(rotation)

    diagonal = []
    for i in range(n_orb):
        if real_rotations:
            diagonal.append(1)
        else:
            diagonal.append(np.exp(1j * params[k]))
            k = k + 1

    return givens_rotations, diagonal


def get_givens_pack_params(givens_rotations, diagonal):
    params = []
    for layer in givens_rotations:
        for gr in layer:
            params.append(gr[2])
            params.append(gr[3])

    for d in diagonal:
        params.append(-1j* np.log(d))

    return params


class HardwareEfficientGivensAnsatz(GenericAnsatz):
    """
    ansatz type: (e^k e^iJ) * n_terms
    spin symmetry alpha = beta

    """
    def __init__(self, hf_reference_fock, givens_rotations, n_terms, init='zeros', local=None, ignore_parity=True, real_rotations=False):
        """

        :param hf_reference_fock: HF reference in fock space
        :param givens_rotations: dictionary of givens rotations to be applied (same format as openfermion)
        :param n_terms: number of layers in ansatz
        :param init: initialization ('zeros', 'ones', 'random')
        :param local: use laocal approximation for J interaction term
        :param ignore_parity: ignore parity between non-adjacent givens rotations
        :param real_rotations: use real rotations
        """

        super().__init__()
        self._operators = []
        self._parameters = []
        self._matrices = []
        self._spin_t1 = None

        self._reference_fock = hf_reference_fock
        self._givens_rotations = givens_rotations

        self._n_terms = n_terms
        n_orb = len(hf_reference_fock) // 2
        self._diagonal = [0]*n_orb
        self._n_qubits = n_orb * 2
        self._local = n_orb if local is None else local
        self._ignore_parity = ignore_parity
        self._real_rotations = real_rotations

        # safety for local
        if self._local > n_orb:
            self._local = n_orb

        n_param_k, n_param_j = self.get_param_size()

        # print('n_param_k:', n_param_k)
        # print('n_param_j:', n_param_j)

        # initialize parameters
        # n_param = n_param_k * ( 1 + (self._n_terms-1)) + n_param_j * (self._n_terms-1)
        n_param = (n_param_k + n_param_j) * self._n_terms

        if init == 'zeros':
            self._parameters = [0.0] * n_param
        elif init == 'ones':
            self._parameters = [1.0] * n_param
        elif init == 'random':
            self._parameters = np.random.rand(n_param)
        else:
            raise ValueError('init must be either zeros, ones, or random')

        self._operators, self._matrices = self._get_matrices(self._parameters)

    @property
    def parameters(self):
        return np.asarray(self._parameters)

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = np.asarray(parameters)
        self._operators, self._matrices = self._get_matrices(self._parameters)

    def get_param_size(self):
        n_orb = len(self._reference_fock) // 2

        n_param_k = 0
        if self._real_rotations:
            for layer in self._givens_rotations:
                n_param_k += len(layer)
        else:
            for layer in self._givens_rotations:
                n_param_k += len(layer)*2

            n_param_k += len(self._diagonal)

        # n_param_j = (n_orb * n_orb + n_orb) // 2 - (n_orb - self._local) * ((n_orb - self._local) + 1) // 2
        n_param_j = ((n_orb * (n_orb + 1)) - (n_orb - self._local) * ((n_orb - self._local) + 1)) // 2

        return n_param_k, n_param_j

    def _get_matrices(self, parameters):

        operators = []

        # bind parameters
        n_param_k, n_param_j = self.get_param_size()
        n_orb = self.n_qubits // 2

        def generator_from_parameters(parameters):
            from scipy.linalg import logm

            givens_rotations, diagonal = get_givens_pack(parameters, self._givens_rotations, n_orb, real_rotations=self._real_rotations)

            U = from_givens_to_U(givens_rotations, diagonal, n_orb)
            return logm(U)

        def symmetric_from_parameters_(parameters, size):

            mat = np.zeros((size, size))
            i, j = np.triu_indices(size)

            mat[i, j] = parameters
            mat[j, i] = parameters

            from scipy.linalg import expm
            return mat

        def symmetric_from_parameters(parameters, size, local):
            mat = np.zeros((size, size))
            offset = 0

            for d in range(local):
                length = size - d
                values = parameters[offset:offset + length]

                i = np.arange(length)
                j = i + d

                mat[i, j] = values
                mat[j, i] = values

                offset += length

            return mat

        # basis change
        pos = 0
        matrices = []

        for _ in range(self._n_terms):

            kappa_i = generator_from_parameters(parameters[pos:n_param_k+pos])

            kappa_spin = -get_spin_matrix(kappa_i)
            ansatz_u = get_ucc_generator(kappa_spin, None, full_amplitudes=True, tolerance=1e-6)
            U_spin = expm(kappa_spin)

            matrices.append(('K', U_spin))
            operators.append(ansatz_u)
            pos += n_param_k

            diag_i = symmetric_from_parameters(parameters[pos:n_param_j+pos], n_orb, self._local)

            j_mat = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)
            for i in range(n_orb):
                for j in range(n_orb):
                    j_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

            spin_jastrow = get_t2_spinorbitals_absolute_full(j_mat, mixed_spin=True)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
            ansatz_j = get_ucc_generator(None, spin_jastrow, full_amplitudes=True)

            # get n_i n_j matrix in spinorbitals
            spin_diagonal = np.zeros((2*n_orb, 2*n_orb))
            for i in range(2*n_orb):
                for j in range(2*n_orb):
                    spin_diagonal[i, j] = spin_jastrow[i, i, j, j].imag

            matrices.append(('J', spin_diagonal))
            operators.append(ansatz_j)
            pos += n_param_j

        return operators, matrices


    def get_preparation_gates(self, simulator):

        if Configuration().mapping == 'jw':
            # this is only for JW mapping (due to givens rotations implementation)

            state_preparation_gates = simulator.get_reference_gates(self._reference_fock)


            #self._operators, self._matrices = self._get_matrices(self._parameters)

            for matrix, operator in zip(self._matrices, self._operators):

                if operator.is_zero():
                    continue

                if matrix[0] == 'K':
                    # implement rotation term
                    rotation_p = matrix[1]
                    state_preparation_gates += simulator.get_rotation_gates(rotation_p,
                                                                            self.n_qubits,
                                                                            separate_spins=True,
                                                                            add_parity=not self._ignore_parity)

                elif matrix[0] == 'J':
                    # implement jastrow term
                    jastrow_mat = matrix[1]
                    state_preparation_gates += simulator.get_density_density_gates(jastrow_mat, self.n_qubits)
                    # jastrow_qubit = operator.get_quibits_list(reorganize=False)
                    # state_preparation_gates += simulator.get_exponential_gates(jastrow_qubit, self.n_qubits)
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

    @property
    def n_qubits(self):
        return len(self._reference_fock)

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
                                       #('H', [4.0, 0.0, 0.0]),
                                       #('H', [6.0, 0.0, 0.0])
                                       ],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='molecule')

    # run classical calculation
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=0, n_orbitals=4, run_ccsd=True)

    n_electrons = molecule.n_electrons
    n_orbitals = molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()


    print('n_electrons', n_electrons)
    print('n_orbitals', n_orbitals)

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, molecule.n_qubits)

    U = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])

    # compatible with openfermion givens_decomposition_square return. Angles are ignored
    givens_rotations, diagonal = givens_decomposition_square(U)

    # 1 layers of givens rotations with 2 rotations:
    # 1) between 0 & 1 orbitals
    # 2) between 2 & 3 orbitals

    givens_rotations = [( (0, 1), (2, 3) )]

    ansatz = HardwareEfficientGivensAnsatz(hf_reference_fock, givens_rotations, n_terms=1, init='random', ignore_parity=False)
    ansatz.print_circuit(simulator)

    print('Energy E: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))
    print('Energy S: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))

    parameters = [0.345, -0.2345, 0.88, 0.4567, -0.43432, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #print(ansatz.parameters)
    #print('Energy: ', ansatz.get_energy(parameters, hamiltonian, None))
    #print('Energy: ', ansatz.get_energy(parameters, hamiltonian, simulator))

    from vqemulti.vqe import vqe
    from vqemulti.optimizers import OptimizerParams

    optimizer = OptimizerParams(method='cobyla')
    results = vqe(hamiltonian, ansatz, energy_simulator=simulator)
    print(results)

    print('Energy E: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))
    print('Energy S: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))
