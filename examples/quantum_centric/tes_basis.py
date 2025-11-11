from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.energy import get_vqe_energy
from openfermion import get_sparse_operator, normal_ordered
from vqemulti.utils import get_sparse_ket_from_fock
from vqemulti.ansatz import get_ucc_ansatz

import numpy as np
import scipy as sp
np.random.seed(42)  # Opcional, per reproduïbilitat


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
            U_test_sp[2*i, 2*j] = U_test[i, j]
            U_test_sp[2*i+1, 2*j+1] = U_test[i, j]

    return U_test_sp

def check_similar(array_1, array_2):
    rel_error = np.linalg.norm(array_1 - array_2) / np.linalg.norm(array_1)
    print(f'check: {rel_error:.5e}   max:{np.max(np.linalg.norm(array_1 - array_2)):.5e} index:{np.argmax(np.linalg.norm(array_1 - array_2))}')


def print_max_values(matrix, tol=1e-2, order=None, spin_notation=False):

    if order is not None:
        matrix = matrix.transpose(order)

    ni, nj = matrix.shape

    for i in range(ni):
        for j in range(nj):
            if abs(matrix[i, j]) > tol:
                if spin_notation is False:
                    print('{:3} {:3}  : {:10.5e}'.format(i, j, matrix[i, j]))
                else:
                    spin = lambda i: 'a' if np.mod(i, 2) == 0 else 'b'
                    print('{:3}{:1} {:3}{:1}: {:10.5e}'.format(i // 2, spin(i),
                                                               j // 2, spin(j),
                                                               matrix[i, j]))


def print_max_values_4d(matrix, tol=1e-2, order=None, spin_notation=False):
    # n_values = len(matrix)

    if order is not None:
        matrix = matrix.transpose(order)

    ni, nj, nk, nl = matrix.shape

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    if abs(matrix[i, j, k, l]) > tol:
                        if spin_notation is False:
                            print('{:3}  {:3}  {:3}  {:3}: {:10.5e}'.format(i, j, k, l, matrix[i, j, k, l]))
                        else:
                            spin = lambda i: 'a' if np.mod(i, 2) == 0 else 'b'
                            print('{:3}{:1} {:3}{:1} {:3}{:1} {:3}{:1}: {:10.5e}'.format(i//2, spin(i),
                                                                                         j//2, spin(j),
                                                                                         k//2, spin(k),
                                                                                         l//2, spin(l),
                                                                                         matrix[i, j, k, l]))



def get_absolute_orbitals(T2):
    """
    T2 to absolute

    :param T2: T2 in [nocc x nocc x nvirt x nvirt] #  a_j a_l a_i^ a_k^
    :return: T2 in [norb x norb x norb x norb] #  a_j a_l a_i^ a_k^
    """
    nocc, nvrt = T2.shape[1:3]
    norb = nocc + nvrt

    T2_abs = np.zeros((norb, norb, norb, norb))

    T2_abs[:nocc, :nocc, nocc:, nocc:] = T2  #  a_j a_l a_i^ a_k^
    # T2_abs = T2_abs.transpose(2, 0, 3, 1) #  a_i^ a_j a_k^ a_l

    return T2_abs


def get_reduced_orbitals(T2_abs, n_occ=1):
    """
    T2 to absolute

    :param T2: T2 in [norb x norb x norb x norb]
    :return: T2 in [nocc x nocc x nvirt x nvirt]
    """
    norb = len(T2_abs)
    nvrt = norb - n_occ

    return T2_abs[:n_occ, :n_occ, n_occ:, n_occ:]



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

    print('**+ inside get_t2_orbitals')
    print(ccsd_double_amps.shape)
    print_max_values_4d(ccsd_double_amps*2)
    print('**')

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


def get_t2_spinorbitals(ccsd_double_amps):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T2 in orbital basis [occupied x occupied x virtual x virtual] (a_i* a_j* a_k a_l)
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] (a_i* a_k a_j* a_l)
    """

    from pyscf.cc.addons import spatial2spin

    norb = sum(ccsd_double_amps.shape)//2

    T2 = spatial2spin(ccsd_double_amps, orbspin=np.array([1, 0]*norb))

    no, nv = T2.shape[1:3]
    nmo = no + nv

    #print('**+ inside get_t2_spinorbitals')
    #print(T2.shape)
    #print_max_values_4d(T2)
    #print('**')

    ccsd_double_amps = np.zeros((nmo, nmo, nmo, nmo), dtype=complex)
    ccsd_double_amps[no:, :no, no:, :no] = .5 * T2.transpose(2, 0, 3, 1)

    return ccsd_double_amps


def get_t1_spinorbitals(ccsd_single_amps):
    """
    get T1 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T1 in orbital basis [occupied x virtual]
    :return: T1 in spinorbitals basis (interleaved) [2nmo x 2nmo] (a_i* a_j)
    """

    from pyscf.cc.addons import spatial2spin

    T1 = spatial2spin(ccsd_single_amps)

    no, nv = T1.shape
    nmo = no + nv

    ccsd_single_amps = np.zeros((nmo, nmo), dtype=complex)
    ccsd_single_amps[no:, :no] = T1.T

    return ccsd_single_amps


def get_t1_spinorbitals_absolute(ccsd_single_amps, n_occ=1):
    """
    get T1 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T1 in orbital basis [nmo x nmo] (a_i* a_j)
    :return: T1 in spinorbitals basis (interleaved) [2nmo x 2nmo] (a_i* a_j)
    """

    from pyscf.cc.addons import spatial2spin

    norb = len(ccsd_single_amps)
    nvrt = norb - n_occ

    ccsd_single_amps_red = ccsd_single_amps[:n_occ, n_occ:]

    T1 = spatial2spin(ccsd_single_amps_red, orbspin=np.array([1, 0] * norb))

    return T1


def get_t2_spinorbitals_absolute(ccsd_double_amps_abs, n_occ=1):
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

    return ccsd_double_amps # a_i^ a_j a_k^ a_l




def get_t2_spinorbitals_absolute_old(ccsd_double_amps_abs, n_occ=1):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_double_amps: T2 in orbital basis [nmo x nmo x nmo x nmo] (a_i* a_j* a_k a_l)
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] (a_i* a_k a_j* a_l)
    """

    ccsd_double_amps = get_reduced_orbitals(ccsd_double_amps_abs, n_occ=n_occ)
    T2 = get_t2_spinorbitals(ccsd_double_amps)

    return T2



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

    # print('check')
    # print_max_values_4d(t2aa)
    #exit()


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




if __name__ == '__main__':

    h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                 ('H', [2.0, 0.0, 0.0]),
                                 #('H', [4.0, 0.0, 0.0]),
                                 #('H', [6.0, 0.0, 0.0])
                                 ],
                       basis='3-21g',
                       multiplicity=1,
                       charge=0,
                       description='molecule')

    # run classical calculation
    molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
    n_total_orb = molecule.n_orbitals
    n_occ = molecule.n_electrons // 2
    print('n_occ: ', n_occ)
    print('n_orb: ', n_total_orb)

    # run CCSD
    ccsd = molecule._pyscf_data.get('ccsd', None)

    print('T2_orb')
    print(ccsd.t2.shape)  # a_i a_j a_a^ a_b^ (occ: i j, vir: a b)
    print(ccsd.t2)

    from pyscf.cc.addons import spin2spatial, spatial2spin

    print('original')
    print(ccsd.t2.shape)
    print_max_values_4d(get_absolute_orbitals(ccsd.t2))


    print('----')
    ccsd_double_amps = spatial2spin(ccsd.t2)
    # ccsd_double_amps = molecule.ccsd_double_amps.transpose(1, 3, 2, 0)*2
    print(ccsd_double_amps.shape)
    print_max_values_4d(get_absolute_orbitals(ccsd_double_amps))
    #print(molecule.ccsd_double_amps)

    #exit()

    # orbspin = np.array([1] * (n_total_orb//2) + [0] * (n_total_orb//2))
    orbspin = np.array([0, 1] * n_total_orb)
    t2aa, t2ab, t2bb = spin2spatial(ccsd_double_amps*0.5, orbspin)
    T2 = t2aa + t2bb + t2ab + t2ab.transpose(1, 0, 3, 2)

    print('recover')
    print(T2.shape)
    print_max_values_4d(T2)

    check_similar(ccsd.t2, T2)

    # exit()

    T1_spin = get_t1_spinorbitals(ccsd.t1)

    print('T1_spin_abs')
    print(T1_spin.shape)
    print_max_values(T1_spin)

    check_similar(T1_spin, molecule.ccsd_single_amps)
    # exit()

    T2_spin = get_t2_spinorbitals(ccsd.t2)

    print('T2_spin_abs')
    print(T2_spin.shape)
    #print_max_values_4d(T2_spin)
    check_similar(T2_spin, molecule.ccsd_double_amps)

    ccsd_t2_rec = get_t2_orbitals(T2_spin, n_occ=n_occ)

    print('T2_orb')
    print(ccsd_t2_rec.shape)
    print(ccsd_t2_rec)

    check_similar(ccsd_t2_rec, ccsd.t2)

    # exit()

    print('\n=======================\n')

    # absolute
    n_orbital = molecule.n_orbitals
    n_virt = n_orbital- n_occ


    print('T2_spin_abs')
    print(molecule.ccsd_double_amps.shape)
    print_max_values_4d(molecule.ccsd_double_amps)

    ccsd_t2_abs = get_t2_orbitals_absolute(molecule.ccsd_double_amps, n_occ=n_occ)

    print('T2_orb_abs')
    print(ccsd_t2_abs.shape)
    print_max_values_4d(ccsd_t2_abs)
    #exit()

    print('T2_orb')
    #ccsd_t2_trans = ccsd_t2_abs.transpose(1, 3, 0, 2)[:n_occ, :n_occ, n_occ:, n_occ:]
    ccsd_t2_trans = get_reduced_orbitals(ccsd_t2_abs, n_occ=n_occ)
    print(ccsd_t2_trans.shape, ccsd.t2.shape)
    print(ccsd_t2_trans)
    check_similar(ccsd_t2_trans, ccsd.t2)

    T2_spin_abs = get_t2_spinorbitals_absolute(ccsd_t2_abs, n_occ=n_occ)

    print('T2_spin_abs')
    print(T2_spin_abs.shape)
    print_max_values_4d(T2_spin_abs)

    check_similar(T2_spin_abs, molecule.ccsd_double_amps)

    #exit()



    # random unitary basis
    print('Test basis orbital and spinorbital')
    from basis_change import random_rotation_qr
    U_test = random_rotation_qr(n_total_orb)

    print('U shape: ', U_test.shape)
    print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))


    # basis in spinorbitals
    U_test_sp = np.zeros((n_total_orb*2, n_total_orb*2), dtype=complex)
    for i in range(n_total_orb):
        for j in range(n_total_orb):
            U_test_sp[2*i, 2*j] = U_test[i, j]
            U_test_sp[2*i+1, 2*j+1] = U_test[i, j]


    # change of basis in orbitals
    T2_orb = get_absolute_orbitals(ccsd.t2)
    T2_orb = T2_orb.transpose(3, 1, 2, 0)
    #print_max_values_4d(T2_orb)

    T2_orb_bc = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test, U_test.conj(), T2_orb, U_test.conj(), U_test)
    print('----')
    # print_max_values_4d(T2_orb_bc)
    # print(np.round(T2_orb_bc, decimals=3)[:2, :2, :2, :2])

    print('full')
    print_max_values_4d(T2_orb)
    print('====')

    #T2_spin_bc_1 = get_t2_spinorbitals_absolute(T2_orb_bc, n_occ=n_occ)
    T2_spin_bc_1 = get_t2_spinorbitals_absolute_full(T2_orb_bc)
    # print_max_values_4d(T2_spin_bc_1)

    print(np.round(T2_spin_bc_1.real*2, decimals=3)[:4, :4, :4, :4])

    print('----')

    # change of basis in spinorbitals
    T2_spin = get_t2_spinorbitals(ccsd.t2)
    # print_max_values_4d(T2_spin.real*2)
    T2_spin_bc_2 = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test_sp, U_test_sp.conj(), T2_spin, U_test_sp.conj(), U_test_sp)

    print(np.round(T2_spin_bc_2.real*2, decimals=3)[:4, :4, :4, :4])

    # print_max_values_4d(T2_spin_bc_2.real*2)

    check_similar(T2_spin_bc_1, T2_spin_bc_2)



