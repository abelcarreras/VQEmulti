from vqemulti.ansatz.generators.rotation import change_of_basis_orbitals
from typing import cast
import scipy
import itertools
import numpy as np


def truncated_eigh(mat, tol=1e-8, max_vecs=None):

    # get eigenvectors
    eigs, vecs = scipy.linalg.eigh(mat)
    if max_vecs is None:
        max_vecs = len(eigs)

    # sort indices
    indices = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[indices]
    vecs = vecs[:, indices]

    # get discarted modes acording to tol
    n_discard = int(np.searchsorted(np.cumsum(np.abs(eigs[::-1])), tol))

    # get final number of modes to keep
    n_vecs = cast(int, min(max_vecs, len(eigs) - n_discard))

    # return remaining eigenvalues and eigenvectors
    return eigs[:n_vecs], vecs[:, :n_vecs]

def quadrature(mat, sign):
    return 0.5 * (1 - sign * 1j) * (mat + sign * 1j * mat.T.conj())


def double_factorized_t2(T2_orb, tol=1e-8, max_vecs=None):
    """
    Double-factorized decomposition of a full-rank t2 tensor of shape (nocc, nocc, nvrt, nvrt).

    :param T2: Full T2 tensor with shape (nocc, nocc, nvrt, nvrt)  a_j a_l a_i^ a_k^
    :param tol: Maximum allowed reconstruction error
    :param max_vecs: Maximum number of singular vectors to keep
    :return: Z, U : array of shape (n_vecs, 2, N, N)
    """

    # get dimensions info
    nocc, nvrt = T2_orb.shape[1:3]
    norb = nocc + nvrt

    # first factorzation
    t2_mat = T2_orb.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)  # a_j a_l a_i^ a_k^ -> (a_j a_i^) x (a_l a_k^) [partial]
    outer_eigs, outer_vecs = truncated_eigh(t2_mat, tol=tol, max_vecs=max_vecs)

    n_vecs = len(outer_eigs)
    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors):
        mat = np.zeros((norb, norb)) # a_j a_i^ [full]
        range_occ = range(nocc)
        range_virt = range(nocc, nocc + nvrt)
        col, row = zip(*itertools.product(range_occ, range_virt)) # recover indices of a_j and a_i^
        mat[row, col] = outer_vec  # a_j a_i^ [full]

        # separate in two hermitian matrices (one_body_tensor[0] + one_body_tensor[1] = mat)
        one_body_tensor[0] = quadrature(mat, sign=1) # a_j a_i^
        one_body_tensor[1] = quadrature(mat, sign=-1) # a_j a_i^

    # 2nd factorization (U^JU)
    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)

    # organize result matrices (absorb eigenvalues in eigenvectors)
    coeffs = np.array([1, -1]) * outer_eigs[:, None] # absorb sign in eigenvalues
    diag_coulomb_mats = (coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]) # N x 2 x n_i n_j (outer product)

    return diag_coulomb_mats, orbital_rotations  # n_i n_j / a_j a_i^


def double_factorized_t2_simple(T2_orb, tol=1e-8, max_vecs=None):
    """
    Double-factorized decomposition of a full-rank t2 tensor of shape (nocc, nocc, nvrt, nvrt).
    This version is more readable but less compact and effcient

    :param T2: Full T2 tensor with shape (nocc, nocc, nvrt, nvrt)  a_j a_l a_i^ a_k^
    :param tol: Maximum allowed reconstruction error
    :param max_vecs: Maximum number of singular vectors to keep
    :return: Z, U : array of shape (n_vecs, 2, N, N)
    """

    # get dimensions info
    nocc, nvrt = T2_orb.shape[1:3]
    norb = nocc + nvrt

    # first factorzation
    t2_mat = T2_orb.transpose(0, 2, 1, 3).reshape(nocc * nvrt, nocc * nvrt)  # a_j a_l a_i^ a_k^ -> (a_j a_i^) x (a_l a_k^) [partial]
    outer_eigs, outer_vecs = truncated_eigh(t2_mat, tol=tol, max_vecs=max_vecs)
    n_vecs = len(outer_eigs)

    diag_coulomb_mats = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    orbital_rotations = np.zeros((n_vecs, 2, norb, norb), dtype=complex)

    for k, outer_vec in enumerate(outer_vecs.T):

        # reshape outer_vec in a matrix
        mat = np.zeros((norb, norb)) # a_j a_i^ [full]
        range_occ = range(nocc)
        range_virt = range(nocc, nocc + nvrt)
        col, row = zip(*itertools.product(range_occ, range_virt)) # recover indices of a_j and a_i^
        mat[row, col] = outer_vec  # a_j a_i^ [full]

        # decompose mat in m_plus and m_minus such that: 1j * m_plus - 1j * m_minus) == mat - mat.T.conj()
        m_plus = quadrature(mat, sign=1)  # a_j a_i^
        m_minus = quadrature(mat, sign=-1)  # a_j a_i^

        assert np.allclose(1j * (m_plus - m_minus), mat - mat.T.conj())
        assert np.allclose(m_plus.conj().T, m_plus)  # check hermitian
        assert np.allclose(m_minus.conj().T, m_minus)  # check hermitian

        # second factorization
        eigs_plus, orb_rot_plus = np.linalg.eigh(m_plus)  # U^ D U
        eigs_minus, orb_rot_minus = np.linalg.eigh(m_minus)  # U^ D U

        # Jastrow matrices J
        diag_coulomb_mats[k, 0, :, :] = + outer_eigs[k] * np.outer(eigs_plus, eigs_plus)  # J = D x D^  (n_i n_j)
        diag_coulomb_mats[k, 1, :, :] = - outer_eigs[k] * np.outer(eigs_minus, eigs_minus)  # J = D x D^  (n_i n_j)

        # Rotation matrices U
        orbital_rotations[k, 0, :, :] = orb_rot_plus  # a_j a_i^
        orbital_rotations[k, 1, :, :] = orb_rot_minus  # a_j a_i^

    return diag_coulomb_mats, orbital_rotations  # n_i n_j / a_j a_i^


