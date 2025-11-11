import numpy as np
import itertools
import scipy
from typing import cast
from tes_basis import print_max_values_4d


def _truncated_eigh(
    mat: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eigs, vecs = scipy.linalg.eigh(mat)
    if max_vecs is None:
        max_vecs = len(eigs)
    indices = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[indices]
    vecs = vecs[:, indices]
    n_discard = int(np.searchsorted(np.cumsum(np.abs(eigs[::-1])), tol))
    n_vecs = cast(int, min(max_vecs, len(eigs) - n_discard))
    return eigs[:n_vecs], vecs[:, :n_vecs]

def _quadrature(mat: np.ndarray, sign: int):
    return 0.5 * (1 - sign * 1j) * (mat + sign * 1j * mat.T.conj())


def double_factorized_t2(
        t2_amplitudes: np.ndarray, *, tol: float = 1e-8,
        max_vecs: int | None = None,
        nocc = 4
) -> tuple[np.ndarray, np.ndarray]:
    r"""Double-factorized decomposition of t2 amplitudes.

    The double-factorized decomposition of a t2 amplitudes tensor :math:`t_{ijab}` is

    .. math::

        t_{ijab} = i \sum_{m=1}^L \sum_{k=1}^2 \sum_{pq}
            Z^{(mk)}_{pq}
            U^{(mk)}_{ap} U^{(mk)*}_{ip} U^{(mk)}_{bq} U^{(mk)*}_{jq}

    Here each :math:`Z^{(mk)}` is a real-valued matrix, referred to as a
    "diagonal Coulomb matrix," and each :math:`U^{(mk)}` is a unitary matrix,
    referred to as an "orbital rotation."

    The number of terms :math:`L` in the decomposition depends on the allowed
    error threshold. A larger error threshold may yield a smaller number of terms.
    Furthermore, the `max_vecs` parameter specifies an optional upper bound
    on :math:`L`. The `max_vecs` parameter is always respected, so if it is
    too small, then the error of the decomposition may exceed the specified
    error threshold.

    Note: Currently, only real-valued t2 amplitudes are supported.

    Args:
        t2_amplitudes: The t2 amplitudes tensor.
        tol: Tolerance for error in the decomposition.
            The error is defined as the maximum absolute difference between
            an element of the original tensor and the corresponding element of
            the reconstructed tensor.
        max_vecs: An optional limit on the number of terms to keep in the decomposition
            of the t2 amplitudes tensor. This argument overrides `tol`.

    Returns:
        - The diagonal Coulomb matrices, as a Numpy array of shape
          `(n_vecs, 2, norb, norb)`.
          The last two axes index the rows and columns of the matrices.
          The first axis indexes the eigenvectors of the decomposition and the
          second axis exists because each eigenvector gives rise to 2 terms in the
          decomposition.
        - The orbital rotations, as a Numpy array of shape
          `(n_vecs, 2, norb, norb)`.
          The last two axes index the rows and columns of the orbital rotations.
          The first axis indexes the eigenvectors of the decomposition and the
          second axis exists because each eigenvector gives rise to 2 terms in the
          decomposition.
    """
    norb, _, _, _ = t2_amplitudes.shape
    # norb = nocc + nvrt

    print('norb:', norb)
    print('nocc:', nocc)

    t2_mat = np.array(t2_amplitudes) # .transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)
    t2_mat = np.array(t2_amplitudes).reshape(norb**2, norb**2)
    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol, max_vecs=max_vecs)
    #print(outer_eigs.shape)
    #print(outer_vecs.shape)


    n_vecs = len(outer_eigs)

    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs, one_body_tensors):
        mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc), range(nocc, norb)))
        mat[row, col] = outer_vec
        one_body_tensor[0] = _quadrature(mat, sign=1)
        one_body_tensor[1] = _quadrature(mat, sign=-1)

    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    coeffs = np.array([1, -1]) * outer_eigs[:, None]
    diag_coulomb_mats = (
        coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    )

    return diag_coulomb_mats, orbital_rotations


import numpy as np
from scipy.linalg import svd

def extract_kron_square_root(M, N):
    """
    Given M = U ⊗ U of shape (N^2 x N^2), try to extract U.
    """
    # Reshape M into a (N^2, N^2) matrix, viewed as a (N, N, N, N) tensor
    M_tensor = M.reshape(N, N, N, N)
    # Permute to group dimensions as (N^2, N^2)
    M_reshaped = np.transpose(M_tensor, (0, 2, 1, 3)).reshape(N*N, N*N)
    # SVD
    U, s, Vh = np.linalg.svd(M_reshaped)
    # First singular vector (best rank-1 approximation)
    u = U[:, 0] * np.sqrt(s[0])
    v = Vh[0, :] * np.sqrt(s[0])
    # Reshape u and v into (N, N)
    U1 = u.reshape(N, N)
    U2 = v.reshape(N, N)
    # Check symmetry: ideally U1 ≈ U2
    if np.allclose(U1, U2, rtol=1.e-5, atol=1.e-8):
        return U1
    else:
        print("Warning: U1 and U2 are not equal — M may not be U ⊗ U.")
        return None


def operator_schmidt(U, n, tol=1e-8):
    """
    Calculate Schmidt decomposition of operator U on C^n ⊗ C^n.
    Returns (lambda_max, A, B) if rank-1, else None.
    """
    U_mat = U.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n*n, n*n)
    # Compute SVD of U_mat as matrix C^(n^2 x n^2)
    # Uu, S, Vh = np.linalg.svd(U_mat, full_matrices=False)
    S, U = np.linalg.eigh(U_mat)
    print(S)

    return U


def double_factorized_t2_general(
    T2: np.ndarray,
    *,
    tol: float = 1e-8,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    N = T2.shape[0] # + T2.shape[2]

    # reshape to (N^2, N^2)
    T2_matrix = T2.reshape(N**2, N**2)
    print(T2_matrix.shape)


    norb = len(T2)
    nocc = 2
    nvrt = norb - nocc


    t2_amplitudes = T2[nocc:, :nocc, nocc:, :nocc]
    print(t2_amplitudes.shape)
    t2_mat = t2_amplitudes.transpose(1, 0, 3, 2).reshape(nocc * nvrt, nocc * nvrt)
    # occ, virt, occ, virt
    print('t2: ', t2_mat.shape)
    print(np.round(t2_mat, decimals=4))



    outer_eigs, outer_vecs = _truncated_eigh(t2_mat, tol=tol, max_vecs=None)
    n_vecs = len(outer_eigs)
    print('outer_eigs', outer_eigs.shape)
    print('outer_vecs', outer_vecs.shape)


    recover_T2_mat = np.zeros((norb, norb), dtype=complex)
    recover_T2_big = np.zeros((norb**2, norb**2), dtype=complex)

    kk = 0
    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb))
        col, row = zip(*itertools.product(range(nocc), range(nocc, nocc + nvrt)))
        mat[row, col] = outer_vec

        one_body_tensor[0] = _quadrature(mat, sign=1)
        one_body_tensor[1] = _quadrature(mat, sign=-1)

        print('hermitian: ', np.allclose(one_body_tensor[0] - one_body_tensor[0].T.conjugate(), 0))
        print('hermitian: ', np.allclose(one_body_tensor[1] - one_body_tensor[1].T.conjugate(), 0))

        A_plus = 0.5 * (1 + 1j) * one_body_tensor[0]
        A_minus = 0.5 * (1 - 1j) * one_body_tensor[1]
        mat_eff = A_plus + A_minus

        print('equal mat: ', np.allclose(mat_eff - mat, 0))

        recover_T2_mat += outer_eigs[kk] * np.outer(outer_vec, outer_vec)

        recover_T2_big += -1j * outer_eigs[kk] * (np.outer(one_body_tensor[0], one_body_tensor[0]) -
                                                 np.outer(one_body_tensor[1], one_body_tensor[1]))

        kk += 1

    error = np.linalg.norm(recover_T2_mat - t2_mat) / np.linalg.norm(t2_mat)
    print(f"Relative reconstruction error: {error:.2e}")
    #exit()

    recover_T2_big = recover_T2_big.reshape((N, N, N, N)).transpose(1, 0, 3, 2)
    #recover_T2_big = np.zeros((N, N, N, N))

    recover_T2_big[:nocc, nocc:, :nocc, nocc:] = 0

    #print(np.round(recover_T2_big.real, decimals=3))
    error = np.linalg.norm(recover_T2_big - T2) / np.linalg.norm(T2)
    print(f"Relative reconstruction error: {error:.2e}")

    print(T2.shape)
    print_max_values_4d(T2.real)

    print('--')
    print_max_values_4d(recover_T2_big.real)

    exit()

    print('*********************')
    recover_T2_big = np.zeros((norb**2, norb**2), dtype=complex)
    for val, mat in zip(outer_eigs, one_body_tensors):
        recover_T2_big += -1j * val * (np.outer(mat[0], mat[0]) -
                                       np.outer(mat[1], mat[1]))

    error = np.linalg.norm(recover_T2_mat - t2_mat) / np.linalg.norm(t2_mat)
    print(f"Relative reconstruction error: {error:.2e}")

    recover_T2_big = recover_T2_big.reshape((N, N, N, N)).transpose(1, 0, 3, 2)
    recover_T2_big[:nocc, nocc:, :nocc, nocc:] = 0

    # Comparació amb la T2 original
    print(np.round(recover_T2_big.real, decimals=3))
    error = np.linalg.norm(recover_T2_big - T2) / np.linalg.norm(T2)
    print(f"Relative reconstruction error: {error:.2e}")

    print('========================')

    print(one_body_tensors.shape)
    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)

    coeffs = np.array([1, -1]) * outer_eigs[:, None]
    diag_coulomb_mats = (
        coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
    )

    return diag_coulomb_mats, orbital_rotations

    print('shape', diag_coulomb_mats.shape)
    print('shape', orbital_rotations.shape)
    print('occ/orb', nocc, norb)

    # T2_rec = reconstruct_T2_from_double_factorization(diag_coulomb_mats, orbital_rotations, nocc, nvrt)
    recover_T2 = np.zeros((norb, norb, norb, norb), dtype=complex)

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
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

    rel_error = np.linalg.norm(recover_T2 - T2) / np.linalg.norm(T2)
    print(f'Relative reconstruction error:: {rel_error:.5e}')



    exit()

    return diag_coulomb_mats, orbital_rotations


def reconstruct_T2_from_double_factorization(
        diag_coulomb_mats: np.ndarray,
        orbital_rotations: np.ndarray,
        nocc: int,
        nvrt: int
) -> np.ndarray:
    """
    Reconstruct T2 tensor from double factorization components.

    Args:
        diag_coulomb_mats: array of shape (n_vecs, 2, norb, norb) - diagonal Coulomb matrices
        orbital_rotations: array of shape (n_vecs, 2, norb, norb) - orbital rotation matrices
        nocc: number of occupied orbitals
        nvrt: number of virtual orbitals

    Returns:
        T2: reconstructed T2 tensor of shape (norb, norb, norb, norb)
    """
    n_vecs = diag_coulomb_mats.shape[0]
    norb = nocc + nvrt

    # Initialize the reconstructed T2 tensor
    T2_reconstructed = np.zeros((norb, norb, norb, norb), dtype=complex)

    # Sum over all factorization vectors
    for vec_idx in range(n_vecs):
        for sign_idx in range(2):  # Two signs in the factorization
            # Get the diagonal Coulomb matrix and orbital rotations for this vector and sign
            diag_coulomb = diag_coulomb_mats[vec_idx, sign_idx]  # shape (norb, norb)
            orbital_rot = orbital_rotations[vec_idx, sign_idx]  # shape (norb, norb)

            # Reconstruct the one-body matrix for this component
            # The one-body matrix is U * diag_coulomb * U^T
            one_body_mat = orbital_rot @ np.diag(diag_coulomb.diagonal()) @ orbital_rot.T

            # Extract the occupied-virtual block (this should correspond to the original mat)
            # The original mat had shape where mat[virt_idx, occ_idx] = outer_vec[occ*nvrt + virt]

            # Add contribution to T2 amplitudes
            # We need to reconstruct the outer product contribution
            for i in range(nocc):
                for a in range(nocc, norb):  # virtual orbitals
                    for j in range(nocc):
                        for b in range(nocc, norb):  # virtual orbitals
                            # The T2 amplitude T2[a,i,b,j] gets contribution from
                            # the outer product of one_body_mat elements
                            T2_reconstructed[a, i, b, j] += one_body_mat[a, i] * one_body_mat[b, j]

    return T2_reconstructed


def change_of_basis(ccsd_single_amps, ccsd_double_amps, U_test):

    n_orb = len(U_test)
    n_qubit = 2* n_orb

    U_test_sp = np.zeros((n_qubit, n_qubit), dtype=complex)

    for i in range(n_orb):
        for j in range(n_orb):
            U_test_sp[2*i, 2*j] = U_test[i, j]
            U_test_sp[2*i+1, 2*j+1] = U_test[i, j]

    # bc_single_amps = U_test.conj().T @ ccsd_single_amps @ U_test
    # T2 (a_i* a_j a_k* a_l)
    bc_single_amps = np.einsum('ai, ab, bj -> ij', U_test_sp.conj(), ccsd_single_amps, U_test_sp)
    bc_double_amps = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test_sp, U_test_sp.conj(), ccsd_double_amps, U_test_sp.conj(), U_test_sp)

    # alternative
    if False:
        n = len(U_test_sp)
        ccsd_double_amps = ccsd_double_amps.transpose(0, 2, 1, 3)
        ccsd_double_amps_rs = ccsd_double_amps.reshape(n**2, n**2)
        U_kron = np.kron(U_test_sp, U_test_sp)
        T_mat_prime = U_kron.conj().T @ ccsd_double_amps_rs @ U_kron
        bc_double_amps = T_mat_prime.reshape(n, n, n, n)
        bc_double_amps = bc_double_amps.transpose(0, 2, 1, 3)

    return bc_single_amps, bc_double_amps, U_test_sp


def print_amplitudes(sparse_cc_state, n_orbitals, n_electrons):

    def tensor(op_list):
        from scipy.sparse import kron

        op = op_list[0]
        for i in range(1, len(op_list)):
            op = kron(op, op_list[i], format='csr')
        return op

    def get_sparse_vector(occupations_vector):
        """
        get sparse vector state  from occupations vector state

        :param occupations_vector: occupations vector state
        :return: sparse vector state
        """

        zero, one = np.array([1, 0]), np.array([0, 1])
        return tensor([one if occupations_vector[i] == 1 else zero for i in range(len(occupations_vector))]).T

    import itertools

    def generate_occupation_vectors(n_electrons, n_orbitals):
        # occupied_positions = itertools.combinations(range(n_orbitals), n_electrons)

        # Generem el vector per cada combinació
        for occupied_positions in itertools.combinations(range(n_orbitals), n_electrons):
            vec = [0] * n_orbitals
            for pos in occupied_positions:
                vec[pos] = 1

            yield vec

    sum = 0
    for basis_vector in generate_occupation_vectors(n_electrons, n_orbitals):
        sparse_basis = get_sparse_vector(basis_vector)
        vev_str = ''.join([str(r) for r in basis_vector])
        ampl = (sparse_basis.getH() @ sparse_cc_state)[0, 0]
        sum += ampl.conjugate() * ampl
        if ampl > 0.05:
            print('{} {:8.4f} {:8.4f}'.format(vev_str, ampl.real, ampl.imag))
    print('sum amplitudes: ', sum)
    exit()

