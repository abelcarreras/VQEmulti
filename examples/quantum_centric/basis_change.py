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


def check_similar(array_1, array_2):
    rel_error = np.linalg.norm(array_1 - array_2) / np.linalg.norm(array_1)
    print(f'check:: {rel_error:.5e}')

def print_max_values(matrix, tol=1e-2):
    n_values = len(matrix)
    for i in range(n_values):
        for j in range(n_values):
            if abs(matrix[i, j]) > tol:
                print('{:3} {:3}  : {:10.5e}'.format(i, j, matrix[i, j]))


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




def random_rotation_qr(n):
    """Genera matriu de rotació correcta amb QR"""
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)

    # Correcció de fase adequada
    d = np.diag(R)
    ph = d / np.abs(d)  # Signes de la diagonal de R
    Q = Q @ np.diag(ph)

    # Si encara té determinant negatiu, corregeix la última columna
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q

def get_sparse_cc(t1, t2, tolerance=1e-15, print_analysis=False):
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

        # print('*** len_op_cc: ', len(operator_tot.terms), sum)

    return get_sparse_operator(operator_tot, n_qubits=n_qubits)


def get_sparse_basis_change_exp(U_test, tolerance=1e-6):
    from openfermion import FermionOperator, get_sparse_operator, hermitian_conjugated
    from scipy.linalg import logm

    n_spin_orbitals = U_test.shape[0]

    kappa = logm(U_test)
    kappa = (kappa - kappa.conj().T) / 2

    assert np.allclose(kappa + kappa.conj().T, 0), "kappa not anti-hermitian!"

    # Construir l’operador de segon quantització
    generator = FermionOperator()
    for p in range(n_spin_orbitals):
        for q in range(p, n_spin_orbitals):
            if abs(kappa[p, q]) > tolerance:
                #term = FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}")
                #generator += kappa[p, q] * term

                generator += kappa[p, q].real * (FermionOperator(f"{p}^ {q}") - FermionOperator(f"{q}^ {p}"))
                generator += 1.0j * kappa[p, q].imag * (FermionOperator(f"{p}^ {q}") + FermionOperator(f"{q}^ {p}"))

    print('len_op_bc: ', len(generator.terms))
    # print('is anti-hermitian? ', normal_ordered(generator + hermitian_conjugated(generator)).isclose(FermionOperator.zero(), tolerance))
    # print('is hermitian? ', normal_ordered(generator - hermitian_conjugated(generator)).isclose(FermionOperator.zero(), tolerance))
    assert normal_ordered(generator + hermitian_conjugated(generator)).isclose(FermionOperator.zero(), tolerance), 'BC not anti-hermitian'

    return get_sparse_operator(generator, n_qubits=n_spin_orbitals)


def get_jastrow(diag_coulomb_mats, orbital_rotations, tolerance=1e-6):
    from openfermion import FermionOperator, get_sparse_operator, hermitian_conjugated
    print(diag_coulomb_mats.shape)
    print(orbital_rotations.shape)

    total = None
    for Z_test, U_test in zip(diag_coulomb_mats, orbital_rotations):

        n_orbitals = U_test.shape[0]
        n_qubit = 2 * n_orbitals

        U_test_sp = np.zeros((n_qubit, n_qubit), dtype=complex)
        Z_test_sp = np.zeros((n_qubit, n_qubit))

        for i in range(n_orbitals):
            for j in range(n_orbitals):
                U_test_sp[2 * i, 2 * j] = U_test[i, j]
                U_test_sp[2 * i + 1, 2 * j + 1] = U_test[i, j]

                # alpha - alpha
                Z_test_sp[2 * i, 2 * j] = Z_test[0, i, j]

                # beta - beta
                Z_test_sp[2 * i + 1, 2 * j + 1] = Z_test[0, i, j]

                # alpha - beta
                Z_test_sp[2 * i, 2 * j+1] = Z_test[1, i, j]

                # beta - alpha
                Z_test_sp[2 * i + 1, 2 * j] = Z_test[1, i, j]

        generator_J = FermionOperator()
        for i in range(n_qubit):
            for j in range(n_qubit):
                if abs(Z_test_sp[i, j]) > tolerance:
                    # print(Z_test_sp[i, j], Z_test_sp[j, i])
                    operator = -1.0j * FermionOperator('{0:}^ {0:} {1:}^ {1:}'.format(i, j))
                    generator_J += Z_test_sp[i, j] * operator

        assert normal_ordered(generator_J + hermitian_conjugated(generator_J)).isclose(FermionOperator.zero(), tolerance), ' J not anti-hermitian'

        print('unitary: ', np.allclose(U_test_sp.conj().T @ U_test_sp, np.eye(U_test_sp.shape[0])))
        # print('Rotation: ', np.linalg.det(U_test) > 0)
        sparse_K = get_sparse_basis_change_exp(U_test_sp)

        sparse_J = get_sparse_operator(generator_J)
        sparse_U = sp.sparse.linalg.expm(-sparse_K)

        if total is None:
            total = sparse_U.getH() @ sp.sparse.linalg.expm(sparse_J) @ sparse_U
        else:
            total = total @ sparse_U.getH() @ sp.sparse.linalg.expm(sparse_J) @ sparse_U
        break

    return total

    # raise Exception('error matrices')


def direct_eigendecomposition_approach(T2_spin, n_reps=1):
    """
    de la matriu de correlacions parcials.
    """

    # recover orbital basis

    T2 = get_t2_orbitals_old(T2_spin)
    print(T2)
    print(T2.shape)
    print_max_values_4d(T2)

    T2 = get_t2_orbitals_absolute(ccsd_double_amps)
    print(T2.shape)
    print_max_values_4d(T2)


    print_max_values_4d(T2)
    print('max: ', np.max(abs(T2)))

    from factor import extract_kron_square_root, double_factorized_t2_general
    diag_coulomb_mats, orbital_rotations = double_factorized_t2_general(T2)

    return diag_coulomb_mats, orbital_rotations

    #from factor import double_factorized_t2
    #diag_coulomb_mats, orbital_rotations = double_factorized_t2(T2, tol=1e-8)

    #print(diag_coulomb_mats.shape)
    #print(orbital_rotations.shape)
    #exit()

    norb = T2.shape[0]


    def build_t2_decomposition(T2: np.ndarray, max_vecs: int):
        N = T2.shape[0]
        T2_mat = T2.reshape(N ** 2, N ** 2)
        U, s, Vh = np.linalg.svd(T2_mat)

        Z = np.zeros((max_vecs, 2, N, N))
        U_rot = np.zeros((max_vecs, 2, N, N))

        for m in range(max_vecs):
            u = U[:, m] * np.sqrt(s[m])
            v = Vh[m, :] * np.sqrt(s[m])

            U1 = u.reshape(N, N)
            U2 = v.reshape(N, N)

            # Ara sí: reconstrueix T2 com a
            # T2 ≈ sum_m U1 ⊗ U2
            Z[m, 0] = np.ones((N, N))  # factor unitari
            Z[m, 1] = np.ones((N, N))
            U_rot[m, 0] = U1
            U_rot[m, 1] = U2

        return Z, U_rot



    print(diag_coulomb_mats.shape)
    print(orbital_rotations.shape)
    print('------')


    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
    diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 4)]
    pairs_aa = None
    pairs_ab = None

    if pairs_aa is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_aa)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 0] *= mask
    if pairs_ab is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_ab)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 1] *= mask

    #print(diag_coulomb_mats)
    #exit()

    #T2_original = recover_original(diag_coulomb_mats, orbital_rotations)

    #print_max_values_4d(T2_spin - T2_original)

    #exit()

    return diag_coulomb_mats, orbital_rotations


def recover_original(diag_coulomb_mats, orbital_rotations):

    print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    nocc = 2

    print('shape', diag_coulomb_mats.shape)
    print('shape', orbital_rotations.shape)
    print('occ/orb', nocc, norb)

    # T2_rec = reconstruct_T2_from_double_factorization(diag_coulomb_mats, orbital_rotations, nocc, nvrt)
    recover_T2 = np.zeros((norb, norb, norb, norb), dtype=complex)

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
        print(U.shape)
        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = 1 * diag[0][i, j]

        print('U.shape: ', U.shape)
        print('z_mat.shape', z_mat.shape)

        # recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[0].conj(), U[0], z_mat, U[0].conj(), U[0])
        recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[0].conj(), U[0], z_mat, U[0], U[0].conj())

        print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = -1 * diag[1][i, j]

        # recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[1], z_mat, U[1].conj(), U[1])
        recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[1], z_mat, U[1], U[1].conj())

        print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

    # recover_T2 = recover_T2.transpose(0, 1, 2, 3)
    recover_T2[:nocc, nocc:, :nocc, nocc:] = 0

    print(np.round(recover_T2, decimals=3).real)

    return recover_T2

def get_t2_orbitals(ccsd_double_amps, n_occ=2):
    """
    get T2 in orbital basis from spinorbitals basis

    :param ccsd_double_amps: T2 in spinorbitals basis (interleaved)
    :return: T2 in orbital basis
    """
    from pyscf.cc.addons import spin2spatial

    norb = len(ccsd_double_amps)//2

    ccsd_double_amps = ccsd_double_amps[n_occ:, :n_occ, n_occ:, :n_occ]
    ccsd_double_amps = ccsd_double_amps.transpose(1, 3, 0, 2) # indices in the origianl order

    # print('**+')
    # print(ccsd_double_amps.shape)
    # print_max_values_4d(ccsd_double_amps)
    # print('**')

    t2aa,t2ab,t2bb = spin2spatial(ccsd_double_amps, np.array([0, 1] * norb))

    return t2aa + t2bb + t2ab + t2ab.transpose(1, 0, 3, 2)


def get_t2_orbitals_absolute(ccsd_double_amps, n_occ=2):


    t2 = get_t2_orbitals(ccsd_double_amps, n_occ=n_occ)
    print('ref', t2.shape)
    print(t2)

    n_occ, _, n_virt, _ = t2.shape

    n_orb = n_occ + n_virt
    t2_full = np.zeros([n_orb, n_orb, n_orb, n_orb])
    t2_full[n_occ:, :n_occ, n_occ:, :n_occ] = t2.transpose(2, 0, 3, 1)


    return t2_full



def get_t2_orbitals_old(ccsd_double_amps):
    """
    get T2 in orbital basis from spinorbitals basis

    :param ccsd_double_amps: T2 in spinorbitals basis (interleaved)
    :return: T2 in orbital basis
    """
    n_orbitals = len(ccsd_double_amps)//2

    T2 = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    #print(ccsd_double_amps[2 * i, 2 * j, 2 * k + 1, 2 * l + 1], ccsd_double_amps[2 * i + 1, 2 * j + 1, 2 * k, 2 * l])
                    T2[i, j, k, l] = ccsd_double_amps[2*i,   2*j,   2*k+1, 2*l+1] + \
                                     ccsd_double_amps[2*i+1, 2*j+1, 2*k,   2*l]

    return T2

import numpy as np

def get_t2_orbitals_alternative(t2_spin):
    """
    Convert T2 from spin-orbital (interleaved) to spatial orbital representation.

    Parameters
    ----------
    t2_spin : ndarray
        T2 amplitudes in spin-orbital basis, shape (2n, 2n, 2n, 2n)

    Returns
    -------
    T2_spatial : ndarray
        T2 amplitudes in spatial orbital basis, shape (n, n, n, n)
    """

    n_so = t2_spin.shape[0]
    n_orb = n_so // 2

    T2 = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=t2_spin.dtype)

    # Mapping indices: α = 0, β = 1 interleaved
    occ_alpha = np.arange(0, n_so, 2)
    occ_beta  = np.arange(1, n_so, 2)
    vir_alpha = np.arange(0, n_so, 2)
    vir_beta  = np.arange(1, n_so, 2)

    for i in range(n_orb):
        for j in range(n_orb):
            for a in range(n_orb):
                for b in range(n_orb):
                    # aa block
                    T2[i,j,a,b] += t2_spin[occ_alpha[i], occ_alpha[j], vir_alpha[a], vir_alpha[b]]
                    # bb block
                    T2[i,j,a,b] += t2_spin[occ_beta[i], occ_beta[j], vir_beta[a], vir_beta[b]]
                    # ab block
                    T2[i,j,a,b] += t2_spin[occ_alpha[i], occ_beta[j], vir_alpha[a], vir_beta[b]]
                    # ba block (apply minus sign if antisymmetry was included in t2_spin)
                    T2[i,j,a,b] += t2_spin[occ_beta[i], occ_alpha[j], vir_beta[a], vir_alpha[b]]


    return T2


def get_t2_spinorbitals(ccsd_double_amps):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis

    :param ccsd_double_amps: T2 in orbital basis [occupied x occupied x virtual x virtual]
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo] (a_i* a_j* a_k a_l)
    """

    from pyscf.cc.addons import spatial2spin

    T2 = spatial2spin(ccsd_double_amps)

    no, nv = T2.shape[1:3]
    nmo = no + nv

    #print('**+ inside get_t2_spinorbitals')
    #print(T2.shape)
    #print_max_values_4d(T2)
    #print('**')

    ccsd_double_amps = np.zeros((nmo, nmo, nmo, nmo))
    ccsd_double_amps[no:, :no, no:, :no] = .5 * T2.transpose(2, 0, 3, 1)

    return ccsd_double_amps


def get_t2_spinorbitals_absolute(ccsd_double_amps, n_occ=1):
    """
    get T2 in spinorbitals basis (interleaved) from orbitals basis (absolute)

    :param ccsd_double_amps: T2 in orbital basis [nmo x nmo x nmo x nmo] (a_i* a_j* a_k a_l)
    :return: T2 in spinorbitals basis (interleaved) [2nmo x 2nmo x 2nmo x 2nmo]
    """

    ccsd_double_amps = ccsd_double_amps.transpose(1, 3, 0, 2)[:n_occ, :n_occ, n_occ:, n_occ:]

    T2 = get_t2_spinorbitals(ccsd_double_amps)

    return T2

if __name__ == '__main__':

    h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                 ('H', [2.0, 0.0, 0.0]),
                                 #('H', [4.0, 0.0, 0.0]),
                                 #('H', [6.0, 0.0, 0.0])
                                 ],
                       basis='sto-3g',
                       multiplicity=1,
                       charge=0,
                       description='molecule')

    # run classical calculation

    #n_frozen_orb = 0
    #n_total_orb = 4
    #molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
    #                     frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)


    molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
    n_total_orb = molecule.n_orbitals
    n_frozen_orb = 0


    ccsd = molecule._pyscf_data.get('ccsd', None)
    print(ccsd.t1.shape)
    print(ccsd.t2.shape)  # a_i a_j a_a^ a_b^ (occ: i j, vir: a b)
    print_max_values_4d(ccsd.t2, order=[2, 1, 3, 0])  #  a_j a_l a_i^ a_k^

    #print_max_values_4d(ccsd.t2)
    print('---')
    spin_t2 = get_t2_spinorbitals(ccsd.t2)
    print_max_values_4d(spin_t2)
    print('===')
    print_max_values_4d(molecule.ccsd_double_amps)

    print('--')
    T2 = get_t2_orbitals(molecule.ccsd_double_amps)
    print(T2)


    print('molecule.ccsd_single_amps')
    print_max_values_4d(molecule.ccsd_double_amps)
    print('-----')
    #print_max_values_4d(get_t2_orbitals(molecule.ccsd_double_amps))


    #dm1 = ccsd.make_rdm1()
    #dm2 = ccsd.make_rdm2()

    # print_max_values_4d(dm2, tol=9e-1)
    # exit()

    hamiltonian = molecule.get_molecular_hamiltonian()
    ccsd_single_amps = molecule.ccsd_single_amps * 0# * 100
    # a_i^ a_j a_k^ a_l
    ccsd_double_amps = molecule.ccsd_double_amps# *0 #*2


    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
    print('energy HF: ', hf_energy)

    sparse_ham = get_sparse_operator(hamiltonian)
    sparse_reference = get_sparse_ket_from_fock(hf_reference_fock)

    print('DIM hamiltonian:', sparse_ham.shape)
    print('DIM reference:', sparse_reference.shape)

    print('N electrons: ', n_electrons)
    print('N orbitals: ', n_orbitals)
    print('N spin-orbitals: ', n_qubits)


    print('HF energy (sparse):')
    print(sparse_reference.T @ sparse_ham @ sparse_reference)


    sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps)
    print('DIM UCC:', sparse_cc_op.shape)


    print('UCC energy (sparse):')
    # spase_cc_state = sp.linalg.expm(sparse_cc_op.toarray()) @ sparse_reference
    sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op, sparse_reference)
    print(sparse_cc_state.T @ sparse_ham @ sparse_cc_state)



    # basis change
    print('\n--------------\n'
          'BASIS CHANGE')

    if False:
        diag_coulomb_mats, orbital_rotations = direct_eigendecomposition_approach(ccsd_double_amps)

        ccsd_double_amps_recover = recover_original(diag_coulomb_mats, orbital_rotations)
        # get_t2_spinorbitals(ccsd_double_amps_recover)


        print('shape_orignal', ccsd_double_amps.shape)
        T2 = get_t2_orbitals_absolute(ccsd_double_amps)
        print('shape_orignal', T2.shape)

        # recover orbital basis

        rel_error = np.linalg.norm(ccsd_double_amps_recover - T2) / np.linalg.norm(T2)
        print(f'Relative reconstruction error:: {rel_error:.5e}')


        print('======= OK ======')
        exit()


        T2_recover_spin = get_t2_spinorbitals_absolute(ccsd_double_amps_recover, n_occ=2)

        rel_error = np.linalg.norm(T2_recover_spin - ccsd_double_amps) / np.linalg.norm(ccsd_double_amps)
        print(f'Relative reconstruction error:: {rel_error:.5e}')

        exit()





        exit()

        print(T2.shape)

        T2 = get_t2_spinorbitals(T2)

        print('original')
        print_max_values_4d(ccsd_double_amps)

        print('T2')
        print_max_values_4d(T2)

        rel_error = np.linalg.norm(T2 - ccsd_double_amps) / np.linalg.norm(ccsd_double_amps)
        print(f'Relative reconstruction error:: {rel_error:.5e}')


        ccsd_double_amps = ccsd_double_amps_recover

        exit()


        # sparse_jastrow = get_jastrow(diag_coulomb_mats, orbital_rotations)

        # print(sparse_jastrow)

        #print('Jastrow energy:')
        #sparse_cc_state = sparse_jastrow @ sparse_reference
        #print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)

        #from factor import print_amplitudes
        #print_amplitudes(sparse_cc_state, n_qubits, n_electrons)

    U_test = random_rotation_qr(n_orbitals)

    print('U shape: ', U_test.shape)
    print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))
    print('Rotation: ', np.linalg.det(U_test) > 0)

    print('U_test')
    print(U_test)

    print('ccsd_double_amps')
    print_max_values_4d(ccsd_double_amps*2, spin_notation=True)

    from factor import change_of_basis
    bc_single_amps, bc_double_amps, U_test_sp = change_of_basis(ccsd_single_amps, ccsd_double_amps, U_test)
    #bc_double_amps = ccsd_double_amps
    print('DIM CC operators')
    print_max_values_4d(bc_double_amps.real * 2, spin_notation=True)


    print('\nanalysis original CC')
    get_sparse_cc(ccsd_single_amps, ccsd_double_amps, print_analysis=True)

    print('analysis basis changed CC')
    get_sparse_cc(ccsd_single_amps, bc_double_amps, print_analysis=True)
    print()

    print(bc_double_amps.shape)

    #print_max_values_4d(bc_double_amps, spin_notation=True)
    #exit()

    # ref
    print('UCC energy (basis change):')
    sparse_cc_op_bc = get_sparse_cc(bc_single_amps, bc_double_amps)
    sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_bc, sparse_reference)
    print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)


    # generate operator that reproduces U_test rotation on CC amplitudes
    sparse_K = get_sparse_basis_change_exp(U_test_sp)
    sparse_U = sp.sparse.linalg.expm(-sparse_K)

    ### np.set_printoptions(linewidth=200)
    ### print(U_test_sp.real)
    ### print(np.kron(U_test_sp.T, U_test_sp).real)
    ### print(sparse_U.toarray().real)



    # simulate basis change of exponent
    sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps)
    sparse_cc_op_bc_2 = sparse_U @ sparse_cc_op @ sparse_U.getH()


    print('UCC energy (reproduce change with sparse_U):')
    sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_bc_2, sparse_reference)
    print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)

    # switch betwen exact and bc
    # sparse_cc_op_bc = sparse_cc_op_bc_2

    #print(sparse_cc_op_bc_2)
    #print(sparse_cc_op_bc)


    if False:
        #print('DIM sparse K', sparse_K.shape)
        print('DIM sparse U', sparse_U.shape)
        print('DIM sparse CC op', sparse_cc_op.shape)

        state_old = sparse_reference
        state_new = sparse_U @ state_old
        state_back = sparse_U.getH() @ state_new

        overlap = np.vdot(state_old.toarray(), state_back.toarray())
        print('Is U unitary? ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0]), atol=1e-10))
        print('Overlap between original and recovered state:', overlap)
        print('Norm of original state:', np.linalg.norm(state_old.toarray()))
        print('Norm of recovered state:', np.linalg.norm(state_back.toarray()))


        #sparse_U = sparse_U.getH()

        print("Is U unitary?", np.allclose(sparse_U.getH().toarray() @ sparse_U.toarray(), np.eye(sparse_U.shape[0])))

        print('basis change U')
        sparse_cc_exp_U = sparse_U @ sp.sparse.linalg.expm(sparse_cc_op) @ sparse_U.getH()
        sparse_cc_state = sparse_cc_exp_U @ sparse_reference
        print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)

        sparse_cc_op_U = sparse_U @ sparse_cc_op @ sparse_U.getH()
        sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_U, sparse_reference)
        print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)


    print('UCC energy (basis change inverse):')

    sparse_cc_exp_U = sparse_U.getH() @ sp.sparse.linalg.expm(sparse_cc_op_bc) @ sparse_U
    sparse_cc_state = sparse_cc_exp_U @ sparse_reference
    print(sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state)

