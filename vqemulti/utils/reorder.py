from openfermion import InteractionOperator
from openfermion.transforms import reorder, get_interaction_operator, get_fermion_operator
import numpy as np


def permute_interaction_operator(hamiltonian, permutation):
    """
    Reorder orbitals of an InteractionOperator.

    :param hamiltonian: InteractionOperator
    :param permutation: list[int]
    :return: InteractionOperator
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

    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = permutation.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder hamiltonian
    if isinstance(hamiltonian, InteractionOperator):
        return permute_interaction_operator(hamiltonian, permutation)
    else:
        return reorder(hamiltonian, order_function)

def permute_amplitudes(T1, T2, permutation):
    """
    Reorder T1 and T2  amplitudes according to a permutation.

    :param T1: 1-e amplitudes absolute basis (N,N)
    :param T2: 2-e amplitudes absolute basis (N,N,N,N)
    :param permutation: array-like
    :return: T1_ord, T2_ord
    """

    p = np.asarray(permutation)

    T1_new = T1[np.ix_(p, p)]
    T2_new = T2[np.ix_(p, p, p, p)]

    return T1_new, T2_new


def permute_reference(hf_reference_fock, permutation):
    return np.array(hf_reference_fock)[list(permutation)].tolist()


def permute_pool(pool, permutation):
    """
    Reorder pool of operators according to a permutation.

    :param pool: operator pool
    :param permutation: permutation
    :return: permuted pool
    """
    from vqemulti.pool.tools import OperatorList

    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = permutation.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder operator pool
    reordered_pool = []
    for op in pool:
        reordered_pool.append(reorder(op, order_function))

    reordered_pool = OperatorList(reordered_pool, normalize=False, antisymmetrize=False, spin_symmetry=False)

    return reordered_pool


def print_permutation(G_qpu, centers, permutation):
    """
    print info about the permutation

    :param G_qpu: connectivity graph
    :param centers: list of centers of orbitals
    :param permutation: list of permutations
    """

    import networkx as nx
    import matplotlib.pyplot as plt

    n_orbitals = len(permutation)

    print('mapping original -> final')

    for i, p in enumerate(permutation):
        print(i, '->', p)

    pos = {permutation[i]: centers[i][:2] for i in range(n_orbitals)}

    permutation_inv = np.argsort(permutation).tolist()

    edge_labels = {}
    for i, j in G_qpu.edges:
        i2 = permutation_inv[i]
        j2 = permutation_inv[j]

        d = np.linalg.norm(centers[i2][:2] - centers[j2][:2])
        edge_labels[(i, j)] = f"{d:.2f}"


    labels = {i: permutation_inv[i] for i in G_qpu.nodes}

    nx.draw(G_qpu, pos, with_labels=False)
    nx.draw_networkx_edge_labels(G_qpu, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G_qpu, pos,labels=labels)
    plt.show()


def optimize_mapping(G_qpu, interaction_matrix):
    """
    optimize permutation according to connectivity graph and interaction_matrix (maximize interaction)
    :param G_qpu: connectivity graph
    :param interaction_matrix: interaction matrix
    :return: permutation, score
    """
    from scipy.optimize import quadratic_assignment
    import networkx as nx


    res = quadratic_assignment(interaction_matrix,
                               nx.to_numpy_array(G_qpu),
                               method="2opt",
                               options={"maximize": True})

    permutation = res.col_ind
    score = res.fun

    return permutation, score

