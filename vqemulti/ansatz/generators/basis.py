import numpy as np


np.random.seed(42)  # Optional for reproducibility


def get_spin_matrix(U_test, dtype=complex):
    """
    get 2D matrix in interleaved spin version

    :param U: rotation matrix in orbitals
    :return: rotation matrix in spin orbitals
    """
    n_orb = len(U_test)
    n_qubit = 2* n_orb

    U_test_sp = np.zeros((n_qubit, n_qubit), dtype=dtype)

    for i in range(n_orb):
        for j in range(n_orb):
            U_test_sp[2*i, 2*j] = U_test[i, j]  # alpha
            U_test_sp[2*i+1, 2*j+1] = U_test[i, j]  # beta

    return U_test_sp


###################################
#  T2 double electrons terms
###################################

def get_reduced_orbitals(T2_abs, n_occ=1):
    """
    T2 to absolute

    :param T2: T2 in [norb x norb x norb x norb]
    :return: T2 in [nocc x nocc x nvirt x nvirt]
    """
    norb = len(T2_abs)
    nvrt = norb - n_occ

    return T2_abs[:n_occ, :n_occ, n_occ:, n_occ:]


def get_t2_absolute_orbitals(T2):
    """
    T2 to absolute

    :param T2: T2 in [nocc x nocc x nvirt x nvirt] #  a_j a_l a_i^ a_k^
    :return: T2 in [norb x norb x norb x norb] #  a_j a_l a_i^ a_k^
    """
    nocc, nvrt = T2.shape[1:3]
    norb = nocc + nvrt

    T2_abs = np.zeros((norb, norb, norb, norb))

    T2_abs[:nocc, :nocc, nocc:, nocc:] = T2  # a_j a_l a_i^ a_k^
    # T2_abs = T2_abs.transpose(2, 0, 3, 1)  # a_i^ a_j a_k^ a_l

    return T2_abs


def get_t2_spinorbitals(ccsd_double_amps):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T2 in orbital basis [occupied x occupied x virtual x virtual] (a_j a_l a_i^ a_k^)
    :return: T2 in spinorbitals basis (interleaved) [2n_mo x 2n_mo x 2n_mo x 2n_mo] (a_i^ a_j a_k^ a_l)
    """

    from pyscf.cc.addons import spatial2spin

    norb = sum(ccsd_double_amps.shape)//2

    T2 = spatial2spin(ccsd_double_amps, orbspin=np.array([1, 0]*norb))

    no, nv = T2.shape[1:3]
    nmo = no + nv

    ccsd_double_amps = np.zeros((nmo, nmo, nmo, nmo), dtype=complex)
    ccsd_double_amps[no:, :no, no:, :no] = .5 * T2.transpose(2, 0, 3, 1)

    return ccsd_double_amps # a_i^ a_j a_k^ a_l


def get_t2_spinorbitals_absolute_old(ccsd_double_amps_abs, n_occ=1):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_double_amps: T2 in orbital basis [nmo x nmo x nmo x nmo] (a_j a_l a_i^ a_k^)
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] (a_i^ a_j a_k^ a_l)
    """

    from pyscf.cc.addons import spatial2spin

    norb = len(ccsd_double_amps_abs)

    ccsd_double_amps = ccsd_double_amps_abs[:n_occ, :n_occ, n_occ:, n_occ:] # a_j a_l a_i^ a_k^
    # print('n_occ: ', n_occ)

    T2 = spatial2spin(ccsd_double_amps, orbspin=np.array([1, 0] * norb)) # a_j a_l a_i^ a_k^ -> a_k^ a_i^ a_l a_j

    nspin_orb = norb * 2
    nspin_occ = n_occ * 2

    ccsd_double_amps = np.zeros((nspin_orb, nspin_orb, nspin_orb, nspin_orb), dtype=complex)
    ccsd_double_amps[nspin_occ:, :nspin_occ, nspin_occ:, :nspin_occ] = .5 * T2.transpose(2, 0, 3, 1) # a_k^ a_i^ a_l a_j ->  a_i^ a_j a_k^ a_l

    return ccsd_double_amps

def get_t2_spinorbitals_absolute(ccsd_double_amps_abs, n_occ=1):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_double_amps: T2 in orbital basis [nmo x nmo x nmo x nmo] (a_j a_l a_i^ a_k^)
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] (a_i^ a_j a_k^ a_l)
    """

    T2_orb = ccsd_double_amps_abs.transpose(2, 0, 3, 1) # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
    ccsd_double_amps = get_t2_spinorbitals_absolute_full(T2_orb)

    return ccsd_double_amps # a_i^ a_j a_k^ a_l



def get_t2_spinorbitals_absolute_full(ccsd_double_amps_abs, dtype=complex):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_double_amps: T2 in orbital basis [nmo x nmo x nmo x nmo] a_i^ a_j a_k^ a_l
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] a_i^ a_j a_k^ a_l
    """

    n_orb = len(ccsd_double_amps_abs)

    T2_spin = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb), dtype=dtype)

    # Create antisymmetrized alpha-alpha amplitude
    # t2aa[i,j,a,b] = t2[i,j,a,b] - t2[j,i,a,b]
    # t2aa = ccsd_double_amps_abs - ccsd_double_amps_abs.transpose(0, 1, 2, 3)
    # t2aa = ccsd_double_amps_abs - ccsd_double_amps_abs.transpose(0, 3, 2, 1)
    t2aa = ccsd_double_amps_abs - ccsd_double_amps_abs.transpose(2, 1, 0, 3)
    # t2aa = ccsd_double_amps_abs - ccsd_double_amps_abs.transpose(2, 3, 0, 1)

    for i in range(n_orb):
        for j in range(n_orb):
            for k in range(n_orb):
                for l in range(n_orb):

                    #        i^(virt)   j(occ)    k^(virt)    l(occ)
                    # alpha, beta
                    T2_spin[2 * i,     2 * j,     2 * k,     2 * l] = t2aa[i, j, k, l] * 0.5
                    T2_spin[2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * l + 1] = t2aa[i, j, k, l] * 0.5

                    # Coulomb
                    T2_spin[2 * i,     2 * j,     2 * k + 1, 2 * l + 1 ] = ccsd_double_amps_abs[i, j, k, l] * 0.5
                    T2_spin[2 * i + 1, 2 * j + 1, 2 * k,     2 * l] = ccsd_double_amps_abs[i, j, k, l] * 0.5

                    # Exchange
                    T2_spin[2 * i,     2 * j + 1, 2 * k + 1, 2 * l] = -ccsd_double_amps_abs[k, j, i, l] * 0.5 #
                    T2_spin[2 * i + 1, 2 * j,     2 * k,     2 * l + 1] = -ccsd_double_amps_abs[k, j, i, l] * 0.5 #

    return T2_spin

def get_t1_spinorbitals_absolute_full(ccsd_single_amps_abs, n_occ=None, dtype=complex):
    """
    get T1 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_single_amps: T1 in orbital basis [nmo x nmo ] a_i^ a_j
    :return: T1 in spinorbitals basis (interleaved) [2nmo x 2nmo ] a_i^ a_j
    """

    n_orb = len(ccsd_single_amps_abs)

    T1_spin = np.zeros((2 * n_orb, 2 * n_orb), dtype=dtype)


    for i in range(n_orb):
        for j in range(n_orb):
            #        i^(virt)   j(occ)
            T1_spin[2 * i,     2 * j] = ccsd_single_amps_abs[i, j] #* 0.5
            T1_spin[2 * i + 1, 2 * j + 1] = ccsd_single_amps_abs[i, j] #* 0.5

    if n_occ is not None:
        T1_spin[:n_occ, n_occ:] = 0

    return T1_spin



###################################
#  T1 single electrons terms
###################################


def get_t1_spinorbitals(ccsd_single_amps):
    """
    get T1 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T1 in orbital basis [occupied x virtual] (a_j a_i^)
    :return: T1 in spinorbitals basis (interleaved) [2nmo x 2nmo] (a_i^ a_j)
    """

    from pyscf.cc.addons import spatial2spin

    T1 = spatial2spin(ccsd_single_amps)

    no, nv = T1.shape
    nmo = no + nv

    ccsd_single_amps = np.zeros((nmo, nmo), dtype=complex)
    ccsd_single_amps[no:, :no] = T1.T

    return ccsd_single_amps  # a_j^ a_i


def get_t1_spinorbitals_absolute(ccsd_single_amps, n_occ=1):
    """
    get T1 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T1 in orbital basis [nmo x nmo] a_j  a_i^
    :return: T1 in spinorbitals basis (interleaved) [2nmo x 2nmo] a_i^  a_j
    """

    from pyscf.cc.addons import spatial2spin

    norb = len(ccsd_single_amps)
    nvrt = norb - n_occ

    ccsd_single_amps_red = ccsd_single_amps[:n_occ, n_occ:]

    T1 = spatial2spin(ccsd_single_amps_red, orbspin=np.array([1, 0] * norb))

    nspin_orb = norb * 2
    nspin_occ = n_occ * 2

    ccsd_single_amps = np.zeros((nspin_orb, nspin_orb), dtype=complex)
    ccsd_single_amps[nspin_occ:, :nspin_occ] = T1.transpose(1, 0) # a_k^ a_i^ a_l a_j ->  a_i^ a_j a_k^ a_l

    return ccsd_single_amps


def get_t1_absolute_orbitals(T1):
    """
    T1 to absolute

    :param T1: T1 in [nocc  x nvirt] #  a_j  a_i^
    :return: T2 in [norb  x norb] #  a_j a_i^
    """
    nocc, nvrt = T1.shape
    norb = nocc + nvrt

    T1_abs = np.zeros((norb, norb))

    T1_abs[:nocc, nocc:] = T1  #  a_j  a_i^


    return T1_abs


###################################
#  Universal
###################################

def get_absolute_orbitals(T_diag):
    """
    from (occupied x virtual) to absolute orbital

    :param T_diag: nocc x nvirt / nocc x nocc x nvirt x nvirt amplitudes
    :return: (n_orb x n_orb) / (n_orb x n_orb x n_orb x n_orb) amplitudes
    """

    if len(np.shape(T_diag)) == 2:
        return get_t1_absolute_orbitals(T_diag)

    elif len(np.shape(T_diag)) == 4:
        return get_t2_absolute_orbitals(T_diag)

    else:
        raise Exception('Inconsistent amplitudes shape')


#####


def get_sparse_basis_change_exp(U_test, tolerance=1e-6):
    """
    get the generator of the unitary transformation of the basis change U_test

    :param U_test: rotation matrix [a_i^ a_j]
    :param tolerance: tolerance
    :return: sparse matrix representation of the generator
    """
    from openfermion import FermionOperator, hermitian_conjugated, get_sparse_operator, normal_ordered
    from scipy.linalg import logm
    from numpy.testing import assert_almost_equal
    import scipy as sp

    n_spin_orbitals = U_test.shape[0]

    kappa = -1. * logm(U_test)  # convention

    assert_almost_equal(U_test, sp.sparse.linalg.expm(-kappa), err_msg='in generator K ')
    assert np.allclose(kappa + kappa.conj().T, 0), "kappa not anti-hermitian!"

    # build operator in second quantization
    generator = FermionOperator()
    for p in range(n_spin_orbitals):
        for q in range(n_spin_orbitals):
            if np.linalg.norm(kappa[p, q]) > tolerance:
                generator += kappa[p, q] * FermionOperator(f"{p}^ {q}")

    assert normal_ordered(generator + hermitian_conjugated(generator)).isclose(FermionOperator.zero(), tolerance), 'BC not anti-hermitian'

    return get_sparse_operator(generator, n_qubits=n_spin_orbitals)


##############
# extra
##############


def get_t2_orbitals(ccsd_double_amps, n_occ=1):
    """
    get T2 in orbital basis from spinorbitals basis

    :param ccsd_double_amps: T2 in spinorbitals basis (interleaved)
    :return: T2 in orbital basis
    """
    from pyscf.cc.addons import spin2spatial

    norb = len(ccsd_double_amps)//2
    n_virt = norb - n_occ
    print(n_occ, n_virt)

    ccsd_double_amps = ccsd_double_amps[n_occ*2:, :n_occ*2, n_occ*2:, :n_occ*2]#[:norb, :norb, :norb, :norb]
    ccsd_double_amps = ccsd_double_amps.transpose(1, 3, 0, 2) # indices in the origianl order


    t2aa, t2ab, t2bb = spin2spatial(ccsd_double_amps, np.array([0, 1] * norb))

    # print(t2aa + t2bb + t2ab + t2ab.transpose(1, 0, 3, 2))

    return t2aa + t2bb + t2ab + t2ab.transpose(1, 0, 3, 2)


def get_t2_orbitals_absolute(ccsd_double_amps, n_occ=1):
    """
    get T2 in orbitals basis (absolute) from spinorbitals basis (interleaved)

    :param ccsd_double_amps: T2 in spinorbitals basis [2norb x 2norb x 2norb x 2norb]
    :param n_occ: occupied orbitals
    :return: T2 in orbitals basis (absolute) [norb x norb x norb x norb] (a_i* a_k a_j* a_l)
    """

    t2 = get_t2_orbitals(ccsd_double_amps, n_occ=n_occ)

    return get_absolute_orbitals(t2)



if __name__ == '__main__':
    from vqemulti.ansatz.generators.rotation import change_of_basis_spin, random_rotation_qr, random_unitary_qr
    import scipy as sp

    n_orbitals = 2

    U_test = random_rotation_qr(n_orbitals)
    U_test = random_unitary_qr(n_orbitals)
    U_test = get_spin_matrix(U_test)

    print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))
    print('Rotation: ', np.linalg.det(U_test) > 0)
    print('U shape: ', U_test.shape)

    print('U_test')
    print(U_test)

    U_test_sp = get_spin_matrix(U_test)

    sparse_K = get_sparse_basis_change_exp(U_test_sp)
    sparse_U = sp.sparse.linalg.expm(sparse_K)

    print('unitary2: ', np.allclose(sparse_U.toarray().conj().T @ sparse_U.toarray(), np.eye(sparse_U.toarray().shape[0])))
    print('Rotation2: ', np.linalg.det(sparse_U.toarray()) > 0)
    print('U shape: ', sparse_U.shape)
    print(np.round(sparse_U.toarray(), decimals=3).real[:6, :6])
    print(np.round(sparse_U.toarray(), decimals=3).imag[:6, :6])
