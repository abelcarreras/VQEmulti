from openfermion import InteractionOperator
from openfermion.transforms import reorder, get_interaction_operator, get_fermion_operator
import numpy as np



def reorder_qubits_adapt(orbitals_order, hamiltonian, hf_reference_fock, pool=None):
    """
    reorder hamiltonian, reference and pool (optional)

    :param orbitals_order: list of indices of the new orbitals order
    :param hamiltonian: hamiltonian operator
    :param hf_reference_fock: reference in Fock space
    :param pool: pool of operators
    :return: reordered Hamiltonian, reference and pool
    """

    n_qubits = len(hf_reference_fock)

    # define reorder function
    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = orbitals_order.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder hamiltonian
    if isinstance(hamiltonian, InteractionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)
        reordered_hamiltonian = reorder(hamiltonian, order_function)
        reordered_hamiltonian = get_interaction_operator(reordered_hamiltonian)
    else:
        reordered_hamiltonian = reorder(hamiltonian, order_function)

    # reorder reference
    reordered_reference = [0] * n_qubits
    for old_idx in range(n_qubits):
        new_idx = order_function(old_idx, n_qubits)
        reordered_reference[new_idx] = hf_reference_fock[old_idx]

    if pool is not None:
        from vqemulti.pool.tools import OperatorList

        # reorder operator pool
        reordered_pool = []
        for op in pool:
            reordered_pool.append(reorder(op, order_function))

        reordered_pool = OperatorList(reordered_pool, normalize=False, antisymmetrize=False, spin_symmetry=False)

        return reordered_hamiltonian, reordered_reference, reordered_pool

    return reordered_hamiltonian, reordered_reference


def reorder_qubits_sqd(orbitals_order, hamiltonian, t2):
    """
    reorder hamiltonian, and T2

    :param orbitals_order: list of indices of the new orbitals order
    :param hamiltonian: hamiltonian operator
    :param t2: t2 amplitudes
    :return: reordered Hamiltonian and t2
    """

    # define reorder function
    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = orbitals_order.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder hamiltonian
    if isinstance(hamiltonian, InteractionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)
        reordered_hamiltonian = reorder(hamiltonian, order_function)
    else:
        reordered_hamiltonian = reorder(hamiltonian, order_function)

    n_occ, n_virt = t2.shape[1:3]

    occ_perm = [orbitals_order.index(i) for i in range(n_occ)]
    virt_perm = [orbitals_order.index(i + n_occ) - n_occ for i in range(n_virt)]

    t2_reordered = t2[np.ix_(occ_perm, occ_perm, virt_perm, virt_perm)]

    return reordered_hamiltonian, t2_reordered


def permute_interaction_operator(hamiltonian, permutation):
    """
    Reorder orbitals of an InteractionOperator.

    Parameters
    ----------
    hamiltonian : InteractionOperator
    permutation : list[int]
        Mapping:
            new_index i corresponds to old_index permutation[i]

    Returns
    -------
    InteractionOperator
    """
    n_orb = len(permutation)

    spin_perm = []
    for p in permutation:
        spin_perm.extend([2 * p, 2 * p + 1])

    constant = hamiltonian.constant

    one_body_old = hamiltonian.one_body_tensor
    two_body_old = hamiltonian.two_body_tensor

    one_body_new = np.zeros_like(one_body_old)
    two_body_new = np.zeros_like(two_body_old)

    for p in range(2*n_orb):
        for q in range(2*n_orb):
            one_body_new[p, q] = one_body_old[spin_perm[p], spin_perm[q]]

    for p in range(2*n_orb):
        for q in range(2*n_orb):
            for r in range(2*n_orb):
                for s in range(2*n_orb):
                    two_body_new[p, q, r, s] = (
                        two_body_old[
                            spin_perm[p],
                            spin_perm[q],
                            spin_perm[r],
                            spin_perm[s]
                        ]
                    )


    return InteractionOperator(constant, one_body_new, two_body_new)

def permute_hamiltonian(hamiltonian, permutation):

    # define reorder function
    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = permutation.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder hamiltonian
    if isinstance(hamiltonian, InteractionOperator):
        # hamiltonian = get_fermion_operator(hamiltonian)
        # return reorder(hamiltonian, order_function)
        return permute_interaction_operator(hamiltonian, permutation)
    else:
        return reorder(hamiltonian, order_function)

def permute_amplitudes(T1, T2, permutation):
    """
    Reorder CC amplitudes according to a permutation.

    Parameters
    ----------
    T1 : ndarray (N,N)
    T2 : ndarray (N,N,N,N)
    permutation : array-like

        permutation[new_index] = old_index

    Returns
    -------
    T1_new, T2_new
    """

    p = np.asarray(permutation)

    T1_new = T1[np.ix_(p, p)]

    T2_new = T2[np.ix_(p, p, p, p)]

    return T1_new, T2_new
