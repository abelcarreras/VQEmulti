from jastrow.factor import double_factorized_t2
from jastrow.basis import get_spin_matrix, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
from jastrow.rotation import change_of_basis_orbitals
from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered
from vqemulti.ansatz.generators import get_ucc_generator
from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.preferences import Configuration
from numpy.testing import assert_almost_equal
import numpy as np
import scipy as sp


def get_basis_change_exp(U_test, tolerance=1e-6, use_qubit=False):
    """
    get the generator of the unitary transformation of the basis change U_test

    :param U_test: rotation matrix [a_i^ a_j]
    :param tolerance: tolerance
    :return: sparse matrix representation of the generator
    """
    from openfermion import FermionOperator, hermitian_conjugated, get_sparse_operator, normal_ordered

    kappa = -1. * sp.linalg.logm(U_test)  # convention

    assert_almost_equal(U_test, sp.sparse.linalg.expm(-kappa), err_msg='in generator K ')
    assert np.allclose(kappa + kappa.conj().T, 0), "kappa not anti-hermitian!"

    return get_ucc_generator(kappa, None, full_amplitudes=True, tolerance=tolerance, use_qubit=use_qubit)


def matrix_power(matrix, exponent):
    """
    only for orthogonal matrices

    :param matrix: orthogonal matrix
    :param exponent: exponent parameter
    :return: matrix power
    """
    # for efficiency and stability of standard UCJ
    if exponent == 1.0:
        return matrix
    if exponent == -1.0:
        return matrix.conj().T

    from scipy.linalg import logm, expm
    return expm(exponent * logm(matrix))


class UnitaryCoupledJastrowAnsatz(ProductExponentialAnsatz):
    """
    ansatz type: e^k e^iJ e^-k
    """
    def __init__(self, t1, t2, full_trotter=True, tolerance=1e-20, use_qubit=False, n_terms=None, local=False):
        """
        assumed HF as reference

        :param t1: single excitations amplitudes matrix (occupied x virtual)
        :param t2: double excitations amplitudes matrix (occupied x occupied x virtual x virtual)
        :param reference_fock: non-entangled initial state as Fock space vector
        """
        #super().__init__()
        self._operators = []
        self._parameters = []
        self._rotation_matrices = []
        self._jastrow_matrices = []
        self._full_trotter = full_trotter

        #t1 = np.array(t1)
        t2 = np.array(t2)
        n_occupied, _, n_virtual, _ = t2.shape
        n_total = n_virtual + n_occupied
        hf_reference_fock = get_hf_reference_in_fock_space(n_occupied*2, n_total*2)


        # assert len(parameters) == len(operator_list)

        #if len(operator_list) > 0 and not is_hermitian(1j * sum(operator_list)):
        #    raise Exception('Non antihermitian operator')


        diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2)  # a_j a_l a_i^ a_k^

        # print(orbital_rotations.shape)
        norb = orbital_rotations.shape[-1]
        n_qubits = norb * 2

        coefficients = []
        operators = []

        if t1 is not None:
            t1_spin = get_t1_spinorbitals(t1)  # a_j a_i^ -> a_i^ a_j
            operator_tot = FermionOperator()
            for i in range(n_qubits):
                for j in range(n_qubits):
                    if abs(t1_spin[i, j]) > tolerance:
                        operator = FermionOperator('{}^ {}'.format(i, j))
                        operator_tot += t1_spin[i, j] * operator - hermitian_conjugated(t1_spin[i, j] * operator)

            coefficients.append(1.0)
            operators.append(operator_tot)

        if n_terms is None:
            n_terms = len(diag_coulomb_mats) * 2

        def make_local(mat):
            local_mat = np.array(mat).copy()
            for i, row in enumerate(local_mat):
                row[i + 2:] = 0
                row[:max(0, i - 1)] = 0
            return local_mat

        i_term = 1
        operator_list = []
        for diag, U in zip(diag_coulomb_mats, orbital_rotations):
            for U_i, diag_i in zip(U, diag):
                if i_term > n_terms:
                    break
                i_term += 1

                if local:
                    # make local version
                    diag_i = np.tril(np.triu(diag_i, -1), 1)

                # diag_i = get_mod_matrix(diag_i, tolerance=7*np.pi/4, fix_antidiagonals=False)

                # build Jastrow operator
                j_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
                for i in range(norb):
                    for j in range(norb):
                        j_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

                if full_trotter:

                    # jastrow
                    spin_jastrow = get_t2_spinorbitals_absolute_full(j_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
                    ansatz_j = get_ucc_generator(None, spin_jastrow, full_amplitudes=True, use_qubit=use_qubit)
                    self._jastrow_matrices.append(ansatz_j)

                    # basis change
                    U_spin = get_spin_matrix(U_i.T)
                    self._rotation_matrices.append(U_spin)
                    ansatz_u = get_basis_change_exp(U_spin, use_qubit=use_qubit)  # a_i^ a_j

                    # add to ansatz
                    coefficients += [-1.0, 1.0, 1.0]
                    operators += ansatz_u
                    operators += ansatz_j
                    operators += ansatz_u

                else:
                    orb_t2 = change_of_basis_orbitals(None, j_mat, U_i.T)[1]  # a_i^ a_j a_k^ a_l
                    spin_t2 = get_t2_spinorbitals_absolute_full(orb_t2)
                    coefficients_c, ansatz_c = get_ucc_generator(None, spin_t2, full_amplitudes=True)

                    coefficients += coefficients_c
                    operators += [op for op in ansatz_c]

        super().__init__(coefficients, operators, hf_reference_fock)

    @property
    def n_qubits(self):
        return len(self._reference_fock)

    @property
    def operators(self):
        return self._operators

    def get_preparation_gates(self, simulator):

        if self._full_trotter and Configuration().mapping == 'jw':
            # this is only for JW mapping (due to givens rotations implementation)

            state_preparation_gates = simulator.get_reference_gates(self._reference_fock)

            i_param = 0
            for rotation, jastrow in zip(self._rotation_matrices, self._jastrow_matrices):

                rotation_param = -self.parameters[i_param]
                rotation_p = matrix_power(rotation, rotation_param)
                state_preparation_gates += simulator.get_rotation_gates(rotation_p, self.n_qubits)

                # implement jastrow term
                jastrow_param = self.parameters[i_param+1]
                jastrow_qubit = jastrow.transform_to_scaled_qubit([jastrow_param])
                state_preparation_gates += simulator.get_exponential_gates(jastrow_qubit, self.n_qubits)

                # implement rotation
                rotation_param = -self.parameters[i_param+2]
                rotation_p = matrix_power(rotation, rotation_param)
                state_preparation_gates += simulator.get_rotation_gates(rotation_p, self.n_qubits)
                i_param += 3

            return state_preparation_gates

        else:
            return super().get_preparation_gates(simulator)


def crop_local_amplitudes(amplitudes, n_neighbors=1):
    """
    remove interactions in the amplitudes between orbitals at larger distance than the set distance

    :param amplitudes: amplitudes matrix (occ x virt) / (occ x occ x virt x virt )
    :param n_neighbors: distance between orbitals
    :return: local amplitudes
    """

    if len(np.shape(amplitudes)) == 2:
        # a_j a_i^
        nocc, nvrt = amplitudes.shape
        local_amplitudes = np.array(amplitudes).copy()  # a_j a_i^
        for i in range(nocc):
            local_amplitudes[i, max(0, n_neighbors - nocc + i+1):] = 0

    elif len(np.shape(amplitudes)) == 4:
        # a_j a_l a_i^ a_k^
        # double electron amplitudes
        nocc, nvrt = amplitudes.shape[1:3]

        local_amplitudes = np.array(amplitudes).copy()  # a_j a_l a_i^ a_k^

        for i in range(nocc):
            # strict criteria, all pairs (i, j) (k, l) within distance
            local_amplitudes[i, :, max(0, n_neighbors - nocc + i + 1):, :] = 0
            local_amplitudes[:, i, :, max(0, n_neighbors - nocc + i + 1):] = 0

    else:
        raise Exception('Amplitudes format not accepted')

    return local_amplitudes


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
    n_frozen_orb = 0 # nothing
    n_total_orb = 4 # total orbitals
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

    n_electrons = molecule.n_electrons - n_frozen_orb * 2

    hamiltonian = molecule.get_molecular_hamiltonian()

    tol_ampl = 0.01

    print('\nJASTROW ansatz\n==============')

    ccsd = molecule._pyscf_data.get('ccsd', None)
    t2 = crop_local_amplitudes(ccsd.t2, n_neighbors=3)
    t1 = ccsd.t1

    ucja = UnitaryCoupledJastrowAnsatz(None, t2, n_terms=2, full_trotter=True)

    energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)

    simulator.print_statistics()
    print(simulator.get_circuits()[-1])

    print('Jastrow energy: ', energy)

    from vqemulti.vqe import vqe
    print(vqe(hamiltonian, ucja, energy_simulator=None))

    energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)
    print('Optimized Jastrow energy: ', energy)

    n_particle = ucja.get_energy(ucja.parameters, n_particles_operator(ucja.n_qubits//2), None)
    spin_z = ucja.get_energy(ucja.parameters, spin_z_operator(ucja.n_qubits//2), None)
    spin_square = ucja.get_energy(ucja.parameters, spin_square_operator(ucja.n_qubits//2), None)

    print('n_particles: {:8.4f}'.format(n_particle))
    print('Sz: {:8.4f}'.format(spin_z))
    print('S2: {:8.4f}'.format(spin_square))


    def get_projections(n_qubits, n_electrons, tolerance=0.01):

        from vqemulti.operators import configuration_projector_operator
        from vqemulti.utils import configuration_generator

        print('\nAmplitudes square')
        amplitudes_square = []
        for configuration in configuration_generator(n_qubits, n_electrons):
            amplitude2 = ucja.get_energy(ucja.parameters, configuration_projector_operator(configuration), None)

            amplitudes_square.append(amplitude2)

            if abs(amplitude2) > tolerance:
                print('{} {:8.5f} '.format(''.join([str(s) for s in configuration]), amplitude2.real))

        print('sum amplitudes square: ', np.sum(amplitudes_square))
        return amplitudes_square

    get_projections(ucja.n_qubits, n_electrons, tolerance=tol_ampl)

    # SQD
    from vqemulti.energy.simulation import simulate_energy_sqd
    energy, samples = simulate_energy_sqd(ucja,
                                          hamiltonian,
                                          simulator_sqd,
                                          n_electrons,
                                          return_samples=True)

    print(samples)
    print('SQD energy', energy)
