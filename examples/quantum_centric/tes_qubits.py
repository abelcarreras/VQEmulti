from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from tes_factor import direct_eigendecomposition_approach, recover_original, get_absolute_orbitals, get_reduced_orbitals
from tes_basis import print_max_values_4d, get_t2_spinorbitals_absolute, get_t2_spinorbitals_absolute, get_t2_spinorbitals
from tes_basis import get_t1_spinorbitals, get_t1_spinorbitals_absolute, get_spin_matrix, print_max_values
from openfermion import get_sparse_operator
from vqemulti.utils import get_sparse_ket_from_fock, get_hf_reference_in_fock_space
import numpy as np
import scipy as sp

np.random.seed(42)  # Opcional, per reproduïbilitat


def get_sparse_cc(t1, t2, tolerance=1e-15, print_analysis=False):
    """
    get sparse representation of UCC operator

    :param t1: amplitudes 1 electron a_i^ a_j
    :param t2: amplitudes 2 electrons a_i^ a_j a_k^ a_l
    :param tolerance:
    :param print_analysis:
    :return:
    """

    from openfermion import FermionOperator, hermitian_conjugated
    n_qubits = len(t1)

    operator_tot = FermionOperator()

    # 1-exciation terms
    for i in range(n_qubits):
        for j in range(n_qubits):
            if abs(t1[i, j]) > tolerance:
                operator = FermionOperator('{}^ {}'.format(i, j))
                operator_tot += t1[i, j] * (operator - hermitian_conjugated(operator))

    # 2-excitation terms
    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(0, n_qubits):  # avoid duplicates
                    if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry
                        if abs(t2[i, j, k, l]) > tolerance:
                            operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                            operator_tot += 0.5 * t2[i, j, k, l] * (operator - hermitian_conjugated(operator))

    if print_analysis:
        # print(operator_tot.terms)
        sum = 0.0
        for i in range(n_qubits):
            for j in range(n_qubits):
                sum += abs(t2[i, j, i, j])

        #print('*** len_op_cc: ', len(operator_tot.terms), sum)

    return get_sparse_operator(operator_tot, n_qubits=n_qubits)



def get_sparse_jastrow_old(diag_coulomb_mats, orbital_rotations, tolerance=1e-15, print_analysis=False):
    from openfermion import FermionOperator, hermitian_conjugated
    from scipy.linalg import logm
    from scipy.sparse import csr_array

    norb = len(orbital_rotations[0][0])
    n_qubits = norb * 2

    sparse_tot = csr_array((2 ** n_qubits, 2 ** n_qubits), dtype=complex)

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):

        # basis change
        U_spin = get_spin_matrix(U[0])
        kappa = logm(U_spin)

        operator_rot = FermionOperator()
        for p in range(n_total_orb):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot += kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot += 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        sparse_rot_1 = sp.sparse.linalg.expm(sparse_rot_exp)

        # Jastrow factor
        operator_jas = FermionOperator()
        for i in range(norb):
            for j in range(norb):
                if abs(diag[0][i, j]) > tolerance:
                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i, 2*i, 2*j, 2*j))
                    operator_jas += -0.5 * diag[0][i, j] * (operator - hermitian_conjugated(operator))

                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i+1, 2*i+1, 2*j+1, 2*j+1))
                    operator_jas += -0.5 * diag[0][i, j] * (operator - hermitian_conjugated(operator))

        sparse_jas = get_sparse_operator(operator_jas, n_qubits=n_qubits)


        operator_rot = FermionOperator()
        for p in range(n_qubits):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot -= kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot -= 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        sparse_rot_2 = sp.sparse.linalg.expm(sparse_rot_exp)


        # print('shape: ', sparse_rot_1.shape, sparse_jas.shape, sparse_rot_2.shape)
        sparse_tot += sparse_rot_1 @ sparse_jas @ sparse_rot_2


       # basis change
        kappa = logm(U[1])
        kappa = get_t1_spinorbitals(kappa)

        operator_rot = FermionOperator()
        for p in range(n_qubits):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot += kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot += 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        sparse_rot_1 = sp.sparse.linalg.expm(sparse_rot_exp)

        # Jastrow factor
        operator_jas = FermionOperator()
        for i in range(norb):
            for j in range(norb):
                if abs(diag[1][i, j]) > tolerance:
                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i, 2*i, 2*j, 2*j))
                    operator_jas += -0.5 * diag[1][i, j] * (operator - hermitian_conjugated(operator))

                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i+1, 2*i+1, 2*j+1, 2*j+1))
                    operator_jas += -0.5 * diag[1][i, j] * (operator - hermitian_conjugated(operator))

        sparse_jas = get_sparse_operator(operator_jas, n_qubits=n_qubits)


        operator_rot = FermionOperator()
        for p in range(n_qubits):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot -= kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot -= 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        sparse_rot_2 = sp.sparse.linalg.expm(sparse_rot_exp)

        sparse_tot += sparse_rot_1 @ sparse_jas @ sparse_rot_2

    return sparse_tot


def recover_original_2(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=2):

    #print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    #print('shape', diag_coulomb_mats.shape)
    #print('shape', orbital_rotations.shape)
    #print('occ/orb', nocc, norb)

    # T2_rec = reconstruct_T2_from_double_factorization(diag_coulomb_mats, orbital_rotations, nocc, nvrt)
    sparse_T2 = sp.sparse.csr_matrix((2**n_qubits, 2**n_qubits), dtype=complex)

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
        # print(U.shape)
        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = 1 * diag[0][i, j]

        # print('U.shape: ', U.shape)
        # print('z_mat.shape', z_mat.shape)

        recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[0].conj(), U[0], z_mat, U[0], U[0].conj())
        # recover_T2 = recover_T2.transpose(3, 1, 2, 0) # a_i^ a_j a_k a_l^ -> a_l a_j a_k^ a_i^ (occ: j l, vir: i k)
        recover_T2 = recover_T2.transpose(0, 2, 1, 3) #  a_i^ a_j a_k a_l^ -> a_i^ a_k^ a_j a_l (occ: j l, vir: i k)

        # print_max_values_4d(recover_T2.real)

        sparse_T2 += get_sparse_cc(ccsd_single_amps*0, get_t2_spinorbitals_absolute(recover_T2, n_occ=n_occ))

        # check hermitian
        assert np.allclose(z_mat - z_mat.T.conjugate(), 0)
        # print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = -1 * diag[1][i, j]

        recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[1], z_mat, U[1], U[1].conj())
        # recover_T2 = recover_T2.transpose(3, 1, 2, 0) # a_i^ a_j a_k a_l^ -> a_l a_j a_k^ a_i^ (occ: j l, vir: i k)
        recover_T2 = recover_T2.transpose(0, 2, 1, 3) #  a_i^ a_j a_k a_l^ -> a_i^ a_k^ a_j a_l (occ: j l, vir: i k)

        #recover_T2[nocc:, nocc:, :nocc, :nocc] = 0
        # print_max_values_4d(recover_T2.real)
        sparse_T2 += get_sparse_cc(ccsd_single_amps*0, get_t2_spinorbitals_absolute(recover_T2, n_occ=n_occ))

        # check hermitian
        assert np.allclose(z_mat - z_mat.T.conjugate(), 0)
        # print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

    return sparse_T2


def recover_original_exp(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=2):
    # print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    #print('shape', diag_coulomb_mats.shape)
    #print('shape', orbital_rotations.shape)
    #print('occ/orb', nocc, norb)

    # T2_rec = reconstruct_T2_from_double_factorization(diag_coulomb_mats, orbital_rotations, nocc, nvrt)
    # sparse_T2 = sp.sparse.csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    from scipy.sparse import identity
    sparse_T2 = identity(2 ** n_qubits, format="csr")


    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
        #print(U.shape)
        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = 1 * diag[0][i, j]

        #print('U.shape: ', U.shape)
        #print('z_mat.shape', z_mat.shape)

        recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[0].conj(), U[0], z_mat, U[0], U[0].conj())
        # recover_T2 = recover_T2.transpose(3, 1, 2, 0)   # a_i^ a_j a_k a_l^ -> a_l a_j a_k^ a_i^ (occ: j l, vir: i k)
        recover_T2 = recover_T2.transpose(0, 2, 1, 3)    # a_i^ a_j a_k^ a_l-> a_i^ a_k^ a_j a_l (occ: j l , vir: i k)

        #recover_T2[nocc:, nocc:, :nocc, :nocc] = 0
        # print_max_values_4d(recover_T2.real)

        recover_T2_sparse = get_sparse_cc(ccsd_single_amps * 0, get_t2_spinorbitals_absolute(recover_T2, n_occ=n_occ))
        sparse_T2 = sparse_T2 @ sp.sparse.linalg.expm(recover_T2_sparse)

        # print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))
        # check hermitian
        assert np.allclose(z_mat - z_mat.T.conjugate(), 0)

        # break

        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = -1 * diag[1][i, j]

        recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[1], z_mat, U[1], U[1].conj())
        #recover_T2 = recover_T2.transpose(3, 1, 2, 0)  # a_l a_j^ a_k a_i^ -> a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
        recover_T2 = recover_T2.transpose(0, 2, 1, 3) # a_i^ a_j a_k^ a_l-> a_i^ a_k^ a_j a_l (occ: j l , vir: i k)
        #recover_T2[nocc:, nocc:, :nocc, :nocc] = 0
        #print_max_values_4d(recover_T2.real)
        recover_T2_sparse = get_sparse_cc(ccsd_single_amps * 0, get_t2_spinorbitals_absolute(recover_T2, n_occ=n_occ))
        sparse_T2 = sparse_T2 @ sp.sparse.linalg.expm(recover_T2_sparse)

        # check hermitian
        assert np.allclose(z_mat - z_mat.T.conjugate(), 0)
        # print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

    return sparse_T2


def recover_original_exp_test(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=2, test_only=False):
    # print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    #print('shape', diag_coulomb_mats.shape)
    #print('shape', orbital_rotations.shape)
    #print('occ/orb', nocc, norb)

    from scipy.sparse import identity
    from tes_basis import get_t2_spinorbitals_absolute_full

    sparse_T2 = identity(2 ** n_qubits, format="csr")

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):

        sign = 1
        for U_i, diag_i in zip(U, diag):
            #U_i = U[0]
            #diag_i = diag[0]


            z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    z_mat[i, i, j, j] = sign * diag_i[i, j]

            #print('U.shape: ', U_i.shape)
            #print('z_mat.shape', z_mat.shape)

            # U_i = np.identity(len(U_i))

            # print_max_values_4d(z_mat)

            if test_only:
                # funciona
                z_mat = z_mat.transpose(2, 0, 3, 1)  # a_k^ a_l a_i^ a_j -> a_i^ a_k^ a_j a_l

                recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U_i.conj(), U_i, z_mat, U_i, U_i.conj())
                # print(recover_T2.shape)

                recover_T2 = recover_T2.transpose(2, 3, 0, 1)  # a_i^ a_k^ a_j a_l -> a_j a_l a_i^ a_k^
                ccsd_double_amps = get_t2_spinorbitals_absolute(recover_T2, n_occ=nocc)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l

                # print_max_values_4d(ccsd_double_amps.real)

            else:

                z_mat = z_mat.transpose(2, 3, 0, 1)  # a_k^ a_l a_i^ a_j ->  a_i^ a_j a_k^ a_l
                ccsd_double_amps = get_t2_spinorbitals_absolute_full(z_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
                ccsd_double_amps = ccsd_double_amps.transpose(0, 2, 1, 3)  # a_i^ a_j a_k^ a_l -> a_i^ a_k^ a_j a_l

                U_spin = get_spin_matrix(U_i)

                recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U_spin.conj(), U_spin, ccsd_double_amps, U_spin, U_spin.conj())
                recover_T2 = recover_T2.transpose(0, 2, 1, 3)  #   a_i^ a_k^ a_j a_l -> a_i^ a_j a_k^ a_l

                spin_occ = n_occ * 2
                ccsd_double_amps = np.zeros_like(recover_T2)
                ccsd_double_amps[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = recover_T2[spin_occ:, :spin_occ, spin_occ:, :spin_occ]


            recover_T2_sparse = get_sparse_cc(ccsd_single_amps * 0, ccsd_double_amps)
            sparse_T2 = sparse_T2 @ sp.sparse.linalg.expm(recover_T2_sparse)

            # check hermitian
            assert np.allclose(z_mat - z_mat.T.conjugate(), 0)
            # print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

            sign = -1

    return sparse_T2



def recover_original_exp_test_2(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, tolerance=1e-15):
    # print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    from scipy.sparse import identity
    from tes_basis import get_t2_spinorbitals_absolute_full

    sparse_T2 = identity(2 ** n_qubits, format="csr")

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):

        sign = 1
        for U_i, diag_i in zip(U, diag):


            z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    z_mat[i, i, j, j] = sign * diag_i[j, i]  # a_i^ a_j a_k^ a_l

            ccsd_double_amps = get_t2_spinorbitals_absolute_full(z_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l

            cb_alternative = False

            if cb_alternative:
                U_spin = get_spin_matrix(U_i.T)
                U_kron = np.kron(U_spin, U_spin)

                ccsd_double_amps_rs = ccsd_double_amps.reshape(n_qubits ** 2, n_qubits ** 2) # (a_i^ a_j)  (a_k^ a_l)
                T_mat_prime = U_kron.conj().T @ ccsd_double_amps_rs @ U_kron
                recover_T2 = T_mat_prime.reshape(n_qubits, n_qubits, n_qubits, n_qubits) # a_i^ a_j a_k^ a_l

            else:
                U_spin = get_spin_matrix(U_i)
                recover_T2 = np.einsum('ia, jb, abcd, kc, ld -> ijkl', U_spin.conj(), U_spin, ccsd_double_amps, U_spin, U_spin.conj())

            spin_occ = n_occ * 2
            ccsd_double_amps = np.zeros_like(recover_T2)
            ccsd_double_amps[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = recover_T2[spin_occ:, :spin_occ, spin_occ:, :spin_occ]

            recover_T2_sparse = get_sparse_cc(ccsd_single_amps * 0, ccsd_double_amps)

            sparse_T2 = sparse_T2 @ sp.sparse.linalg.expm(recover_T2_sparse)

            # check hermitian
            #assert np.allclose(z_mat - z_mat.T.conjugate(), 0)

            sign = -1
            break
        break

    return sparse_T2


def recover_original_exp_test_3(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, tolerance=1e-15, nocc=2):
    # print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    from scipy.sparse import identity
    from tes_basis import get_t2_spinorbitals_absolute_full
    from basis_change import get_sparse_basis_change_exp

    sparse_T2 = identity(2 ** n_qubits, format="csr")

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):

        sign = 1
        for U_i, diag_i in zip(U, diag):

            z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    z_mat[i, i, j, j] = sign * diag_i[j, i]  # a_i^ a_j a_k^ a_l

            ccsd_double_amps = get_t2_spinorbitals_absolute_full(z_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l

            #print_max_values(basis_rot_sparse @ basis_rot_sparse_inv)
            #exit()

            U_spin = get_spin_matrix(U_i)

            if True:
                U_kron = np.kron(U_spin, U_spin)

                ccsd_double_amps_rs = ccsd_double_amps.reshape(n_qubits ** 2, n_qubits ** 2) # (a_i^ a_j)  (a_k^ a_l)
                T_mat_prime = U_kron.conj().T @ ccsd_double_amps_rs @ U_kron
                recover_T2 = T_mat_prime.reshape(n_qubits, n_qubits, n_qubits, n_qubits) # a_i^ a_j a_k^ a_l


            spin_occ = nocc * 2

            ccsd_double_amps = np.zeros_like(recover_T2)
            ccsd_double_amps[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = recover_T2[spin_occ:, :spin_occ, spin_occ:, :spin_occ]

            recover_T2_sparse = get_sparse_cc(ccsd_single_amps * 0, ccsd_double_amps)


            if False:
                sparse_K = get_sparse_basis_change_exp(U_spin) # a_i^ a_j
                basis_rot_sparse = sp.sparse.linalg.expm(sparse_K) # a_i^ a_j
                basis_rot_sparse_inv = sp.sparse.linalg.expm(-sparse_K) # a_i^ a_j
                recover_T2_sparse = basis_rot_sparse_inv @ recover_T2_sparse @ basis_rot_sparse # a_i^ a_j


            sparse_T2 = sparse_T2 @ sp.sparse.linalg.expm(recover_T2_sparse)

            # check hermitian
            assert np.allclose(z_mat - z_mat.T.conjugate(), 0)

            sign = -1
            break
        break

    return sparse_T2




def recover_original_exp_test_4(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, tolerance=1e-15, nocc=2):

    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    print('shape', diag_coulomb_mats.shape)
    print('shape', orbital_rotations.shape)
    print('occ/orb', nocc, norb)

    # T2_rec = reconstruct_T2_from_double_factorization(diag_coulomb_mats, orbital_rotations, nocc, nvrt)
    # sparse_T2 = sp.sparse.csr_matrix((2 ** n_qubits, 2 ** n_qubits), dtype=complex)
    from scipy.sparse import identity
    from basis_change import get_sparse_basis_change_exp
    from openfermion import FermionOperator, hermitian_conjugated
    from tes_basis import get_t2_spinorbitals_absolute_full

    sparse_T2 = identity(2 ** n_qubits, format="csr")

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):

        sign = 1
        for U_i, diag_i in zip(U, diag):

            U_spin = get_spin_matrix(U_i)

            sparse_K = get_sparse_basis_change_exp(U_spin)

            z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    z_mat[i, i, j, j] = sign * diag_i[j, i]  # a_i^ a_j a_k^ a_l

            print('z_mat')
            print_max_values_4d(z_mat.real)

            z_mat_spin_full = get_t2_spinorbitals_absolute_full(z_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l

            spin_occ = n_occ * 2
            z_mat_spin = np.zeros_like(z_mat_spin_full)
            z_mat_spin[spin_occ:, :spin_occ, spin_occ:, :spin_occ] = z_mat_spin_full[spin_occ:, :spin_occ, spin_occ:, :spin_occ]


            print('ccsd_double_amps')
            print_max_values_4d(ccsd_double_amps.real)
            #exit()

            basis_rot_sparse = sp.sparse.linalg.expm(sparse_K)
            basis_rot_sparse_inv = sp.sparse.linalg.expm(-sparse_K)

            print('z_mat')
            # print_max_values_4d(z_mat_spin, spin_notation=True)

            from openfermion import FermionOperator, hermitian_conjugated

            operator_jas = FermionOperator()

            # 2-excitation terms
            for i in range(n_qubits):
                for j in range(n_qubits):
                    for k in range(n_qubits):
                        for l in range(0, n_qubits):  # avoid duplicates
                            if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry
                                if abs(z_mat_spin[i, j, k, l]) > tolerance:
                                    operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                                    operator_jas += 0.5 * z_mat_spin[i, j, k, l] * (operator - hermitian_conjugated(operator))


            # print(operator_jas)
            jas_sparse = get_sparse_operator(operator_jas, n_qubits=n_qubits)

            RJR_sparse = basis_rot_sparse_inv @ sp.sparse.linalg.expm(jas_sparse) @ basis_rot_sparse

            sparse_T2 = sparse_T2 @ RJR_sparse

    return sparse_T2



def get_sparse_jastrow(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, tolerance=1e-15, nocc=2):
    from openfermion import FermionOperator, hermitian_conjugated
    from scipy.linalg import logm

    print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    print('shape', diag_coulomb_mats.shape)
    print('shape', orbital_rotations.shape)
    print('occ/orb', nocc, norb)

    from scipy.sparse import identity
    sparse_T2 = identity(2 ** n_qubits, format="csr")


    for diag, U in zip(diag_coulomb_mats, orbital_rotations):


        U_spin = get_spin_matrix(U[0])
        kappa = logm(U_spin)

        operator_rot = FermionOperator()
        for p in range(n_total_orb):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot += 0.5 * kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot += 0.5j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        basis_rot_sparse = sp.sparse.linalg.expm(sparse_rot_exp)

        # Jastrow factor
        operator_jas = FermionOperator()
        for i in range(norb):
            for j in range(norb):
                if abs(diag[0][i, j]) > tolerance:
                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i, 2*i, 2*j, 2*j))
                    operator_jas += -0.5 * diag[0][i, j] * (operator - hermitian_conjugated(operator))

                    operator = FermionOperator('{}^ {} {}^ {}'.format(2*i+1, 2*i+1, 2*j+1, 2*j+1))
                    operator_jas += -0.5 * diag[0][i, j] * (operator - hermitian_conjugated(operator))

        jas_sparse = get_sparse_operator(operator_jas, n_qubits=n_qubits)
        jas_sparse = sp.sparse.linalg.expm(jas_sparse)


        operator_rot = FermionOperator()
        for p in range(n_qubits):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot -= kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot -= 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        basis_rot_sparse_inv = sp.sparse.linalg.expm(sparse_rot_exp)


        block_sparse = basis_rot_sparse @ jas_sparse @ basis_rot_sparse_inv

        sparse_T2 = sparse_T2 @ block_sparse

        return sparse_T2
        exit()


        U_spin = get_spin_matrix(U[1])
        kappa = logm(U_spin)

        operator_rot = FermionOperator()
        for p in range(n_total_orb):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot += kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot += 1.0j * kappa[p, q].imag * (
                                FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        basis_rot_sparse = sp.sparse.linalg.expm(sparse_rot_exp)

        # Jastrow factor
        operator_jas = FermionOperator()
        for i in range(norb):
            for j in range(norb):
                if abs(diag[0][i, j]) > tolerance:
                    operator = FermionOperator('{}^ {} {}^ {}'.format(2 * i, 2 * i, 2 * j, 2 * j))
                    operator_jas += -0.5 * diag[1][i, j] * (operator - hermitian_conjugated(operator))

                    operator = FermionOperator('{}^ {} {}^ {}'.format(2 * i + 1, 2 * i + 1, 2 * j + 1, 2 * j + 1))
                    operator_jas += -0.5 * diag[1][i, j] * (operator - hermitian_conjugated(operator))

        jas_sparse = get_sparse_operator(operator_jas, n_qubits=n_qubits)
        jas_sparse = sp.sparse.linalg.expm(jas_sparse)

        operator_rot = FermionOperator()
        for p in range(n_qubits):
            for q in range(p, n_qubits):
                if abs(kappa[p, q]) > tolerance:
                    # term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                    # generator += kappa[p, q] * term

                    operator_rot -= kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                    operator_rot -= 1.0j * kappa[p, q].imag * (
                                FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

        sparse_rot_exp = get_sparse_operator(operator_rot, n_qubits=n_qubits)
        basis_rot_sparse_inv = sp.sparse.linalg.expm(sparse_rot_exp)

        block_sparse = basis_rot_sparse @ jas_sparse @ basis_rot_sparse_inv

        sparse_T2 = sparse_T2 @ block_sparse

    return sparse_T2


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

    # parameters
    n_total_orb = molecule.n_orbitals
    n_occ = molecule.n_electrons // 2
    n_electrons = molecule.n_electrons
    n_qubits = n_total_orb * 2

    print('n_occ: ', n_occ)
    print('n_total_orb: ', n_total_orb)

    # hamiltonian
    hamiltonian = molecule.get_molecular_hamiltonian()
    sparse_ham = get_sparse_operator(hamiltonian)
    print('DIM hamiltonian:', sparse_ham.shape)

    # hartree fock
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)
    sparse_reference = get_sparse_ket_from_fock(hf_reference_fock)
    print('DIM reference:', sparse_reference.shape)
    print('HF energy (sparse):', sparse_reference.T @ sparse_ham @ sparse_reference)

    # run CCSD
    ccsd = molecule._pyscf_data.get('ccsd', None)
    print('T2_orb')
    print(ccsd.t2.shape) # a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
    print_max_values_4d(ccsd.t2)

    # a_i^ a_j
    ccsd_single_amps = molecule.ccsd_single_amps * 0  # * 100
    # a_i^ a_j a_k^ a_l
    # ccsd_double_amps = molecule.ccsd_double_amps  # *0 #*2
    ccsd_double_amps = get_t2_spinorbitals(ccsd.t2)

    # UCC
    sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps)
    print('DIM UCC:', sparse_cc_op.shape)

    sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op, sparse_reference)
    print('UCC energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)

    # get jastrow data
    diag_coulomb_mats, orbital_rotations = direct_eigendecomposition_approach(ccsd.t2)

    # recovered
    if True:
        ccsd_double_amps_recover = recover_original(diag_coulomb_mats, orbital_rotations, nocc=n_occ)
        ccsd_double_amps_recover_spin = get_t2_spinorbitals_absolute(ccsd_double_amps_recover, n_occ=n_occ)
        sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps_recover_spin)
        print('DIM recovered:', sparse_cc_op.shape)
        sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op, sparse_reference)
        print('UCC recovered energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)

    # recovered 2  e^(T1+T2)
    sparse_cc_op = recover_original_2(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=n_occ)
    print('DIM recovered 2:', sparse_cc_op.shape)
    sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op, sparse_reference)
    print('UCC recovered 2 energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)
    #exit()

    # recovered exponential e^(T1) * e^(T2)
    sparse_cc_op_exp = recover_original_exp(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=n_occ)
    print('DIM recovered trotter:', sparse_cc_op_exp.shape)
    sparse_cc_state = sparse_cc_op_exp @ sparse_reference
    print('UCC recovered trotter energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)
    #exit()

    # recovered exponential e^(R J1 R^-1) * e^(R J2 R^-1) | Ti = R Ji R^-1
    sparse_cc_op_exp = recover_original_exp_test(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=n_occ)
    print('DIM recovered trotter test:', sparse_cc_op_exp.shape)
    sparse_cc_state = sparse_cc_op_exp @ sparse_reference
    print('**** UCC recovered trotter test energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)

    # exit()
    # recovered exponential
    sparse_cc_op_exp = recover_original_exp_test_2(diag_coulomb_mats, orbital_rotations, ccsd_single_amps)
    print('DIM recovered test 2:', sparse_cc_op_exp.shape)
    sparse_cc_state = sparse_cc_op_exp @ sparse_reference
    print('**** UCC recovered test 2 energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)

    # exit()
    # recovered exponential
    sparse_cc_op_exp = recover_original_exp_test_3(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=n_occ)
    print('DIM recovered test 3:', sparse_cc_op_exp.shape)
    sparse_cc_state = sparse_cc_op_exp @ sparse_reference
    print('**** UCC recovered test 3 energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)

    exit()
    # Jastrow
    #sparse_jastrow_op = get_sparse_jastrow(diag_coulomb_mats, orbital_rotations, ccsd_single_amps, nocc=n_occ)
    #print('DIM Jastrow:', sparse_jastrow_op.shape)
    #sparse_cc_state = sparse_jastrow_op @ sparse_reference
    #print('Jastrow energy (sparse):', sparse_cc_state.T @ sparse_ham @ sparse_cc_state)
