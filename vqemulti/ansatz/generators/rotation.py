from vqemulti.ansatz.generators.basis import get_spin_matrix
import numpy as np


def change_of_basis_orbitals(ccsd_single_amps, ccsd_double_amps, U_test, alternative=False):
    """
    perform a change of basis at orbital basis

    :param ccsd_single_amps: single UCC amplitudes in spinorbitals  #  a_i^ a_j
    :param ccsd_double_amps: double UCC amplitudes in spinorbitals  #  a_i^ a_j a_k^ a_l
    :param U_test: rotation matrix in orbitals basis # a_i^ a_j
    :param alternative: use alternative algorithm for change of basis
    :return: bc_single_amps, bc_double_amps (a_i^ a_j / a_i^ a_j a_k^ a_l)
    """

    if ccsd_single_amps is not None:
        bc_single_amps = np.einsum('ai, ab, bj -> ij', U_test.conj(), ccsd_single_amps, U_test)
    else:
        bc_single_amps = None

    if ccsd_double_amps is not None:
        bc_double_amps = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test.conj(), U_test, ccsd_double_amps, U_test.conj(), U_test)  # a_i^ a_j a_k^ a_l

    else:
        bc_double_amps = None

    #alternative = True
    # alternative (only double)
    if alternative:
        # print('alternative')

        n_qubit = len(U_test)
        U_kron = np.kron(U_test, U_test)

        ccsd_double_amps = ccsd_double_amps.transpose(0, 2, 1, 3)  # a_i^ a_j a_k^ a_l -> a_i^ a_k^ a_j a_l

        ccsd_double_amps_rs = ccsd_double_amps.reshape(n_qubit**2, n_qubit**2) #  a_i^ a_k^ a_j a_l
        T_mat_prime = U_kron.T.conj() @ ccsd_double_amps_rs @ U_kron  #  a_i^ a_k^ a_j a_l
        bc_double_amps = T_mat_prime.reshape(n_qubit, n_qubit, n_qubit, n_qubit) #  a_i^ a_k^ a_j a_l

        bc_double_amps = bc_double_amps.transpose(0, 2, 1, 3)  # a_i^ a_k^ a_j a_l -> a_i^ a_j a_k^ a_l

    return bc_single_amps, bc_double_amps  # a_i a_j^/ a_i^ a_j a_k^ a_l


def change_of_basis_spin(ccsd_single_amps, ccsd_double_amps, U_test, alternative=False):
    """
    perform a change of basis at spinorbital basis (interleaved)

    :param ccsd_single_amps: single UCC amplitudes in spinorbitals  #  a_i^ a_j
    :param ccsd_double_amps: double UCC amplitudes in spinorbitals  #  a_i^ a_j a_k^ a_l
    :param U_test: rotation matrix in orbitals basis
    :param alternative: use alternative algorithm for change of basis
    :return: bc_single_amps, bc_double_amps (a_i^ a_j/ a_i^ a_j a_k^ a_l)
    """

    U_test_sp = get_spin_matrix(U_test)

    bc_single_amps, bc_double_amps  = change_of_basis_orbitals(ccsd_single_amps,
                                                               ccsd_double_amps,
                                                               U_test_sp,
                                                               alternative=alternative)

    return bc_single_amps, bc_double_amps  # a_i^ a_j / a_i^ a_j a_k^ a_l


def random_rotation_qr(n):
    """Generate a rotation matrix using QR"""
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)

    # phase correction
    d = np.diag(R)
    ph = d / np.abs(d)  # diagonal signs of R
    Q = Q @ np.diag(ph)

    # If negative determinant invert last row
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q


def random_unitary_qr(n):
    """Generate a random unitary matrix using complex QR decomposition"""
    # Generate a random complex matrix
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # QR decomposition
    Q, R = np.linalg.qr(A)

    # Phase correction: make diagonal elements of R have unit magnitude
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)

    # Ensure determinant has unit magnitude (unitary group U(n))
    detQ = np.linalg.det(Q)
    Q /= detQ ** (1 / n)  # normalize global phase so det(Q)=1 if you want SU(n)

    return Q