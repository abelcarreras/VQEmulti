from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered
from vqemulti.pool.tools import OperatorList
import numpy as np
from vqemulti.ansatz.generators.ucc import get_ucc_generator
from numpy.testing import assert_almost_equal
import scipy as sp


def jastrow_to_qubits(t2, tolerance=1e-6):
    """
    Get unitary coupled cluster ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: define UCC ansatz in qubit operators
    :return: coefficients as list and ansatz as OperatorsList
    """

    operator_tot = QubitOperator()

    # 2-excitation terms
    n_spin_orbitals = len(t2)
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
                if abs(t2[i, i, j, j]) > tolerance:
                    operator = QubitOperator(())  # identity
                    operator -= QubitOperator(f'Z{i}')
                    operator -= QubitOperator(f'Z{j}')
                    operator += QubitOperator(f'Z{i} Z{j}')
                    operator_tot += 0.25 * t2[i, i, j, j] * operator

    # assert normal_ordered(operator_tot + hermitian_conjugated(operator_tot)).isclose(FermionOperator.zero(), tolerance)

    operators = [operator_tot]
    coefficients = [1.0]

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz



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



def get_mod_matrix(matrix, tolerance=2*np.pi, fix_antidiagonals=True):
    n_orb = len(matrix)
    matrix_mod = np.array(matrix)

    if matrix_mod[0, 0] < matrix_mod[n_orb-1, n_orb-1]:
        tolerance *= -1

    matrix_mod[0, 0] = tolerance
    matrix_mod[n_orb-1, n_orb-1] = -tolerance

    if fix_antidiagonals:
        matrix_mod[n_orb-1, 0] = 0.0
        matrix_mod[0, n_orb-1] = 0.0

    return matrix_mod


def get_ucj_generator(t2, t1=None, full_trotter=True, tolerance=1e-20, use_qubit=False, n_terms=None, local=False):
    """
    Get unitary coupled Jastrow ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param full_trotter: full trotter
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: return operators as QubitOperators
    :param n_terms: number of terms (U^JU) to be included in the ansatz
    :param local: use local version of J operator
    :return: coefficients as list and ansatz as OperatorsList
    """
    from jastrow.factor import double_factorized_t2
    from jastrow.basis import get_spin_matrix, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
    from jastrow.rotation import change_of_basis_orbitals

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
                if abs(t1_spin[i,j]) > tolerance:
                    operator = FermionOperator('{}^ {}'.format(i, j))
                    operator_tot += t1_spin[i, j] * operator - hermitian_conjugated(t1_spin[i, j] * operator)

        coefficients.append(1.0)
        operators.append(operator_tot)

    if n_terms is None:
        n_terms = len(diag_coulomb_mats) * 2

    def make_local(mat):
        local_mat = np.array(mat).copy()
        for i, row in enumerate(local_mat):
            row[i+2:] = 0
            row[:max(0, i-1)] = 0
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

            #diag_i = get_mod_matrix(diag_i, tolerance=7*np.pi/4, fix_antidiagonals=False)

            # build Jastrow operator
            j_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    j_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

            if full_trotter:

                # jastrow
                spin_jastrow = get_t2_spinorbitals_absolute_full(j_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
                coefficients_j, ansatz_j = get_ucc_generator(None, spin_jastrow, full_amplitudes=True, use_qubit=use_qubit)

                # basis change
                U_spin = get_spin_matrix(U_i.T)
                coefficients_u, ansatz_u = get_basis_change_exp(U_spin, use_qubit=use_qubit)  # a_i^ a_j

                # add to ansatz
                coefficients += [-c for c in coefficients_u]
                operators += [op for op in ansatz_u]

                coefficients += coefficients_j
                operators += [op for op in ansatz_j]

                coefficients += [c for c in coefficients_u]
                operators += [op for op in ansatz_u]

            else:
                orb_t2 = change_of_basis_orbitals(None, j_mat, U_i.T)[1]  # a_i^ a_j a_k^ a_l
                spin_t2 = get_t2_spinorbitals_absolute_full(orb_t2)
                coefficients_c, ansatz_c = get_ucc_generator(None, spin_t2, full_amplitudes=True)

                coefficients += coefficients_c
                operators += [op for op in ansatz_c]

    #from utils import get_operators_order
    #ordering = get_operators_order(operators)
    #print('ordering: ', ordering)
    #exit()
    #operators = np.array(operators)[ordering]
    #coefficients = np.array(coefficients)[ordering]

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz
