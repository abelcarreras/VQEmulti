from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.energy import get_vqe_energy
from openfermion import get_sparse_operator, normal_ordered
from vqemulti.utils import get_sparse_ket_from_fock
from vqemulti.ansatz import get_ucc_ansatz
from basis_change import get_sparse_basis_change_exp

import numpy as np
import scipy as sp
np.random.seed(42)  # Opcional, per reproduïbilitat
from tes_basis import *
from basis_change import get_sparse_cc

if __name__ == '__main__':

    h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                 ('H', [2.0, 0.0, 0.0]),
                                 ('H', [4.0, 0.0, 0.0]),
                                 ('H', [6.0, 0.0, 0.0])
                                 ],
                       basis='sto-3g',
                       multiplicity=1,
                       charge=0,
                       description='molecule')

    # run classical calculation
    molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
    n_total_orb = molecule.n_orbitals
    n_qubits = n_total_orb * 2
    n_occ = molecule.n_electrons // 2
    spin_occ = n_occ * 2

    ccsd_double_amps = molecule.ccsd_double_amps

    print('n_occ: ', n_occ)
    print('n_orb: ', n_total_orb)

    # run CCSD
    ccsd = molecule._pyscf_data.get('ccsd', None)

    print('original')
    print(ccsd.t2.shape)
    T1_spin = get_t1_spinorbitals(ccsd.t1)  #  a_j a_i^

    T2_orb = get_absolute_orbitals(ccsd.t2)  #  a_j a_l a_i^ a_k^ ->  a_j a_l a_i^ a_k^
    print('T2_orb')
    print_max_values_4d(T2_orb, order=[2, 0, 3, 1])   #  a_i^ a_j a_k^ a_l

    ccsd_double_amps_alt = get_t2_spinorbitals_absolute(T2_orb, n_occ=n_occ)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
    print_max_values_4d(ccsd_double_amps_alt.real, spin_notation=True)
    check_similar(ccsd_double_amps, ccsd_double_amps_alt)
    #exit()

    # random unitary basis
    print('Test basis orbital and spinorbital')
    from basis_change import random_rotation_qr
    U_test = random_rotation_qr(n_total_orb)
    # U_test = np.identity(n_total_orb)
    print('U_test')
    print(U_test)

    print('U shape: ', U_test.shape)
    print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))

    U_test_sp = get_spin_matrix(U_test.T)
    print('U_sp shape: ', U_test_sp.shape)
    print('unitary: ', np.allclose(U_test_sp.conj().T @ U_test_sp, np.eye(U_test_sp.shape[0])))

    T2_orb = T2_orb.transpose(2, 0, 3, 1)  #  a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l

    #  a_i^ a_j a_k^ a_l
    bc_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U_test.conj(), U_test, T2_orb, U_test, U_test.conj())
    bc_T2 = bc_T2.transpose(1, 3, 0, 2)  # a_i^ a_j a_k^ a_l -> a_j a_l a_i^ a_k^
    bc_T2_spin = get_t2_spinorbitals_absolute(bc_T2, n_occ=n_occ)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l


    from factor import change_of_basis
    _, bc_double_amps, U_test_sp = change_of_basis(ccsd_double_amps_alt[0, 0], ccsd_double_amps_alt, U_test)

    print('DIM CC operators')
    print_max_values_4d(bc_double_amps.real * 2, spin_notation=True)
    check_similar(bc_T2_spin, bc_double_amps)

    exit()

    # change of basis in orbitals
    T2_orb = T2_orb.transpose(0, 2, 1, 3)

    #  a_j a_l a_i^ a_k^
    T2_orb_bc = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test, U_test.conj(), T2_orb, U_test.conj(), U_test)

    #T2_orb = T2_orb.transpose(2, 3, 0, 1) # a_j a_l a_i^ a_k^ ->  a_i ^ a_k ^ a_j a_l
    # a_i ^ a_k ^ a_j a_l
    #T2_orb_bc = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U_test.conj(), U_test, T2_orb, U_test, U_test.conj())
    # a_i^ a_k^ a_j a_l -> a_j a_l a_i^ a_k^
    #T2_orb_bc = T2_orb_bc.transpose(2, 3, 0, 1)

    print('reference')
    T2_spin_bc_1 = get_t2_spinorbitals_absolute(T2_orb_bc, n_occ=n_occ) # a_j a_l a_i^ a_k^ ->  a_i^ a_j a_k^ a_l
    print_max_values_4d(T2_spin_bc_1.real * 2, spin_notation=True)
    exit()


    #print('----')
    #exit()

    # change of basis in spinorbitals
    #T2_spin = get_t2_spinorbitals(ccsd.t2)
    T2_orb = get_absolute_orbitals(ccsd.t2) #  a_j a_l a_i^ a_k^

    if False:
        T2_spin = get_t2_spinorbitals_absolute(T2_orb, n_occ=n_occ) # a_j a_l a_i^ a_k^ ->  a_i^ a_j a_k^ a_l
        T2_spin = T2_spin.transpose(0, 2, 1, 3)  # a_i^ a_j a_k^ a_l -> a_i^ a_k^ a_j a_l
    else:
        T2_orb = T2_orb.transpose(2, 1, 3, 0) # a_j a_l a_i^ a_k^ ->  a_i^ a_j a_k^ a_l
        T2_spin = get_t2_spinorbitals_absolute_full(T2_orb) # a_i^ a_j a_k^ a_l -> a_k^ a_j a_i^ a_j
        T2_spin = T2_spin.transpose(0, 2, 1, 3)  #  a_i^ a_j a_k^ a_l -> a_i^ a_k^ a_j a_l

    # print('before basis change')
    # print_max_values_4d(T2_spin.real * 2, order=[0, 2, 1, 3], spin_notation=True)

    print('check molecule.ccsd_double_amps')
    check_similar(molecule.ccsd_double_amps, T2_spin.transpose(0, 2, 1, 3))

    #print_max_values_4d(T2_spin.real*2)

    # a_i^ a_k^ a_j a_l
    T2_spin_bc_2 = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test_sp, U_test_sp.conj(), T2_spin, U_test_sp.conj(), U_test_sp)
    # T2_spin_bc_2 = T2_spin

    T2_spin_bc_2 = T2_spin_bc_2.transpose(0, 2, 1, 3)  #   a_i^ a_k^ a_j a_l -> a_i^ a_j a_k^ a_l
    # T2_spin_bc_2 = T2_spin_bc_2.transpose(0, 3, 2, 1)  # a_i^ a_k^ a_j a_l -> a_i^ a_l a_k^ a_j

    if True:
        spin_occ = n_occ*2
        T2_spin_bc_2_trunc = np.zeros_like(T2_spin_bc_2)
        T2_spin_bc_2_trunc[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = T2_spin_bc_2[spin_occ:, :spin_occ, spin_occ:, :spin_occ]
        T2_spin_bc_2 = T2_spin_bc_2_trunc

    # print(np.round(T2_spin_bc_2.real*2, decimals=3)[:4, :4, :4, :4])
    print('target')
    print_max_values_4d(T2_spin_bc_2.real * 2, spin_notation=True)

    check_similar(T2_spin_bc_1, T2_spin_bc_2)





    # basis change

    print('----')

    # change of basis in spinorbitals

    T2_orb = get_absolute_orbitals(ccsd.t2) #  a_j a_l a_i^ a_k^
    T2_orb = T2_orb.transpose(2, 1, 3, 0) # a_j a_l a_i^ a_k^ ->  a_i^ a_j a_k^ a_l

    T2_spin = get_t2_spinorbitals_absolute_full(T2_orb) # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l


    # change of basis

    cb_alternative = False
    if cb_alternative:
        U_kron = np.kron(U_test_sp, U_test_sp)
        ccsd_double_amps_rs = T2_spin.reshape(n_qubits ** 2, n_qubits ** 2)  # (a_i^ a_j)  (a_k^ a_l)
        T_mat_prime = U_kron.conj().T @ ccsd_double_amps_rs @ U_kron
        T2_spin_bc_2 = T_mat_prime.reshape(n_qubits, n_qubits, n_qubits, n_qubits)  # a_i^ a_j a_k^ a_l

        T2_spin_bc_2_trunc = np.zeros_like(T2_spin_bc_2)
        T2_spin_bc_2_trunc[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = T2_spin_bc_2[spin_occ:, :spin_occ, spin_occ:, :spin_occ]
        T2_spin_bc_2 = T2_spin_bc_2_trunc

        print('alternative')
        check_similar(T2_spin_bc_1, T2_spin_bc_2)

        T2_sparse_2 = get_sparse_cc(T1_spin * 0, T2_spin_bc_2)


    else:
        sparse_K = get_sparse_basis_change_exp(U_test_sp)
        sparse_U = sp.sparse.linalg.expm(-sparse_K)

        #T2_orb = T2_orb.transpose(1, 3, 0, 2)
        T2_spin = get_t2_spinorbitals_absolute(T2_orb)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l

        #T2_spin_bc_2_trunc = np.zeros_like(T2_spin) #  a_i^ a_j a_k^ a_l
        #T2_spin_bc_2_trunc[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = T2_spin[spin_occ:, :spin_occ, spin_occ:, :spin_occ]
        #T2_spin = T2_spin_bc_2_trunc

        # print_max_values_4d(T2_spin, spin_notation=True)

        # T2_spin = T2_spin.transpose(3, 0, 1, 2) #  a_k^ a_j a_i^ a_j

        sparse_cc_op = get_sparse_cc(T1_spin*0.0, T2_spin)

        T2_sparse_2 = sparse_U.getH() @ sparse_cc_op @ sparse_U


    T2_sparse_1 = get_sparse_cc(T1_spin * 0, T2_spin_bc_1)

    # print(np.round(T2_spin_bc_2.real*2, decimals=3)[:4, :4, :4, :4])
    print('sparse')
    print(' shape', T2_sparse_1.shape)
    # print_max_values_4d(T2_spin_bc_2.real * 2, spin_notation=True)

    print('sparse 1')
    #print_max_values(T2_sparse_1)

    print('sparse 2')
    #print_max_values(T2_sparse_2)

    check_similar(T2_sparse_1.toarray(), T2_sparse_2.toarray())




