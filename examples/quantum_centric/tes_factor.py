from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from tes_basis import print_max_values_4d, get_t2_spinorbitals, get_reduced_orbitals, get_absolute_orbitals

import numpy as np
np.random.seed(42)  # Opcional, per reproduïbilitat




def double_factorized_t2(
    T2: np.ndarray,
    tol: float = 1e-8,
    max_vecs: int | None = None):
    """
    Double-factorized decomposition of a full-rank t2 tensor of shape (N, N, N, N).

    Args:
        T2: Full T2 tensor with shape (N, N, N, N)
        tol: Maximum allowed reconstruction error.
        max_vecs: Maximum number of singular vectors to keep.

    Returns:
        - Z: array of shape (n_vecs, 2, N, N)
        - U: array of shape (n_vecs, 2, N, N)
    """

    from factor import _truncated_eigh, _quadrature
    import itertools

    t2_amplitudes = T2

    nocc, _, nvrt, _ = t2_amplitudes.shape
    norb = nocc + nvrt
    N = norb

    t2_mat = t2_amplitudes.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)

    # occ, virt, occ, virt
    print('t2: ', t2_mat.shape)


    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol, max_vecs=None)
    n_vecs = len(outer_eigs)
    print('outer_eigs', outer_eigs.shape)
    print('outer_vecs', outer_vecs.shape)

    recover_T2_mat = np.zeros((nocc*nvrt, nocc*nvrt), dtype=complex)
    recover_T2_abs = np.zeros((norb**2, norb**2), dtype=complex)

    kk = 0
    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc), range(nocc, nocc + nvrt)))
        mat[row, col] = outer_vec

        one_body_tensor[0] = _quadrature(mat, sign=1)
        one_body_tensor[1] = _quadrature(mat, sign=-1)

        #print('hermitian: ', np.allclose(one_body_tensor[0] - one_body_tensor[0].T.conjugate(), 0))
        #print('hermitian: ', np.allclose(one_body_tensor[1] - one_body_tensor[1].T.conjugate(), 0))

        A_plus = 0.5 * (1 + 1j) * one_body_tensor[0]
        A_minus = 0.5 * (1 - 1j) * one_body_tensor[1]
        mat_eff = A_plus + A_minus

        assert np.allclose(mat_eff - mat, 0)

        recover_T2_mat += outer_eigs[kk] * np.outer(outer_vec, outer_vec)

        recover_T2_abs += -1j * outer_eigs[kk] * (np.outer(one_body_tensor[0], one_body_tensor[0]) -
                                                  np.outer(one_body_tensor[1], one_body_tensor[1]))

        kk += 1


    error = np.linalg.norm(recover_T2_mat - t2_mat) / np.linalg.norm(t2_mat)
    print(f"Relative reconstruction error: {error:.2e}")

    recover_T2_abs = recover_T2_abs.reshape((norb, norb, norb, norb)).transpose(0, 2, 1, 3)  # i* j k* l -> i* k* j l
    recover_T2_abs[nocc:, nocc:, :nocc, :nocc] = 0

    print(recover_T2_abs.shape)
    #print_max_values_4d(recover_T2_big.real)

    T2_abs = get_absolute_orbitals(T2)
    print(T2_abs.shape)
    #print_max_values_4d(T2_abs)

    error = np.linalg.norm(recover_T2_abs - T2_abs) / np.linalg.norm(T2)
    print(f"Relative reconstruction error: {error:.2e}")

    print('*********************')
    recover_T2_abs = np.zeros((norb**2, norb**2), dtype=complex)
    for val, mat in zip(outer_eigs, one_body_tensors):
        recover_T2_abs += -1j * val * (np.outer(mat[0], mat[0]) -
                                       np.outer(mat[1], mat[1]))

    error = np.linalg.norm(recover_T2_mat - t2_mat) / np.linalg.norm(t2_mat)
    print(f"Relative reconstruction error: {error:.2e}")


    recover_T2_abs = recover_T2_abs.reshape((N, N, N, N)).transpose(0, 2, 1, 3)  # a_l a_j^ a_k a_i^ -> a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
    recover_T2_abs[nocc:, nocc:, :nocc, :nocc] = 0

    # comparison with original T2 (absolute)
    T2_abs = get_absolute_orbitals(T2)

    # print(np.round(recover_T2_abs.real, decimals=3))
    error = np.linalg.norm(recover_T2_abs - T2_abs) / np.linalg.norm(T2_abs)
    print(f"Relative reconstruction error: {error:.2e}")


    print('========================')

    print(one_body_tensors.shape)
    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)

    coeffs = np.array([1, -1]) * outer_eigs[:, None]
    diag_coulomb_mats = (
        coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    )

    return diag_coulomb_mats, orbital_rotations


def direct_eigendecomposition_approach(T2_orb, n_reps=1):
    """
    de la matriu de correlacions parcials.
    """

    # recover orbital basis


    diag_coulomb_mats, orbital_rotations = double_factorized_t2(T2_orb)



    return diag_coulomb_mats, orbital_rotations


def recover_original(diag_coulomb_mats, orbital_rotations, nocc=2):

    print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]

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

        recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[0].conj(), U[0], z_mat, U[0], U[0].conj())

        print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

        z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
        for i in range(norb):
            for j in range(norb):
                z_mat[i, i, j, j] = -1 * diag[1][i, j]

        recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[1], z_mat, U[1], U[1].conj())

        print('hermitian: ', np.allclose(z_mat - z_mat.T.conjugate(), 0))

    recover_T2 = recover_T2.transpose(3, 1, 2, 0)  # a_l a_j^ a_k a_i^ -> a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
    recover_T2[nocc:, nocc:, :nocc, :nocc] = 0

    #print(np.round(recover_T2, decimals=3).real)
    print_max_values_4d(recover_T2.real)

    return recover_T2


def recover_original_alt(diag_coulomb_mats, orbital_rotations, nocc=2):

    print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]

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
                z_mat[i, i, j, j] = 1 * (diag[0][i, j] - diag[1][i, j])

        print('U.shape: ', U.shape)
        print('z_mat.shape', z_mat.shape)

        recover_T2 += np.einsum('ia, jb, abcd, kc, ld -> ijkl', U[1].conj(), U[0], z_mat, U[1], U[0].conj())

    recover_T2 = recover_T2.transpose(3, 1, 2, 0)  # a_l a_j^ a_k a_i^ -> a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
    recover_T2[nocc:, nocc:, :nocc, :nocc] = 0

    #print(np.round(recover_T2, decimals=3).real)
    print_max_values_4d(recover_T2.real)

    return recover_T2


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
    n_occ = molecule.n_electrons // 2
    print('n_occ: ', n_occ)
    print('n_total_orb: ', n_total_orb)

    # run CCSD
    ccsd = molecule._pyscf_data.get('ccsd', None)

    print('T2_orb')
    print(ccsd.t2.shape) # a_i^ a_j^ a_a a_b (occ: i j, vir: k l)
    print(ccsd.t2)

    diag_coulomb_mats, orbital_rotations = direct_eigendecomposition_approach(ccsd.t2)

    print('final')
    ccsd_double_amps_recover = recover_original(diag_coulomb_mats, orbital_rotations, nocc=n_occ)

    print('shape_original', ccsd.t2.shape)
    T2 = get_absolute_orbitals(ccsd.t2)

    print('shape_absolute', T2.shape)

    rel_error = np.linalg.norm(ccsd_double_amps_recover - T2) / np.linalg.norm(T2)
    print(f'Relative reconstruction error (absolute): {rel_error:.5e}')

    ccsd_t2_reduced = get_reduced_orbitals(ccsd_double_amps_recover, n_occ=n_occ)

    rel_error = np.linalg.norm(ccsd.t2 - ccsd_t2_reduced) / np.linalg.norm(ccsd_t2_reduced)
    print(f'Relative reconstruction error (reduced): {rel_error:.5e}')

    T2_spin = get_t2_spinorbitals(ccsd_t2_reduced.real)

    rel_error = np.linalg.norm(molecule.ccsd_double_amps - T2_spin) / np.linalg.norm(T2_spin)
    print(f'Relative reconstruction error (spin): {rel_error:.5e}')
