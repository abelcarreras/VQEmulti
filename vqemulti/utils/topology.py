import numpy as np
import networkx as nx
from scipy.optimize import quadratic_assignment


def _score(A, B, perm):
    """sum_ij A[i, j] * B[perm[i], perm[j]] (same as scipy's res.fun)"""
    return np.sum(A * B[np.ix_(perm, perm)])


def _swap_gains(A, B, perm):
    """
    Exact change of the score when the nodes of orbitals i and j are swapped, for all
    pairs at once (A and B must be symmetric). Element [i, j] of the returned matrix.
    """
    Bp = B[np.ix_(perm, perm)]
    C = A @ Bp
    dC, dA, dB = np.diag(C), np.diag(A), np.diag(Bp)

    gain = 2 * (C + C.T - dC[:, None] - dC[None, :]
                - (dA[:, None] - A) * (Bp - dB[:, None])
                - (A - dA[None, :]) * (dB[None, :] - Bp))
    gain += (dA[:, None] - dA[None, :]) * (dB[None, :] - dB[:, None])
    np.fill_diagonal(gain, 0)
    return gain


def _tabu_search(A, B, perm, n_iter, rng):
    """
    Robust tabu search over pair swaps. An orbital may not go back to a node it left
    during the last ~N iterations, unless that move gives a new best score (aspiration).

    :return: best permutation found
    """
    n = len(A)
    current = perm.copy()
    current_score = _score(A, B, current)
    best, best_score = current.copy(), current_score

    tabu = np.zeros((n, n), dtype=int)  # tabu[i, node]: orbital i may not return to node before this iteration
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    tenure = (int(0.9 * n), int(1.1 * n) + 2)

    for it in range(1, n_iter + 1):
        gain = _swap_gains(A, B, current)

        # swap (i, j) is tabu if both orbitals would return to nodes they recently left
        T = tabu[:, current]  # T[i, j] = tabu[i, current[j]]
        forbidden = (T > it) & (T.T > it)
        aspiration = current_score + gain > best_score + 1e-12
        allowed = upper & (~forbidden | aspiration)
        if not allowed.any():
            continue

        i, j = np.unravel_index(np.argmax(np.where(allowed, gain, -np.inf)), gain.shape)
        tabu[i, current[i]] = it + rng.integers(*tenure)
        tabu[j, current[j]] = it + rng.integers(*tenure)
        current[i], current[j] = current[j], current[i]
        current_score += gain[i, j]

        if current_score > best_score + 1e-12:
            best, best_score = current.copy(), current_score

    return best


def optimize_mapping(G_qpu, interaction_matrix, n_restarts=15, n_iter=None, seed=None,
                     initial_guess=None, verbose=False):
    """
    Optimize the orbital -> graph node (qubit) permutation that maximizes the interaction
    between orbitals placed on connected nodes of G_qpu:

        score = sum_ij interaction_matrix[i, j] * adjacency[permutation[i], permutation[j]]

    Each restart: random 'faq' start (scipy) + robust tabu search. Returns the best of all
    restarts (and of initial_guess, if given).

    :param G_qpu: connectivity graph with nodes 0..N-1 (a DiGraph works as its undirected version)
    :param interaction_matrix: (N, N) symmetric matrix (larger value = should be connected)
    :param n_restarts: number of random restarts (defaults take ~1 s for 20 orbitals, ~9 s for 72).
                       If verbose shows the best score reached by only 1 run, increase n_restarts or n_iter
    :param n_iter: tabu iterations per restart (default 100*N)
    :param seed: random seed (int, numpy Generator or None) for reproducible results
    :param initial_guess: optional permutation (orbital i -> node initial_guess[i]) used
                          as the start of one extra tabu search
    :param verbose: print the best score and how many runs reached it
    :return: permutation, score (permutation[i] is the node assigned to orbital i)
    """
    interaction_matrix = np.asarray(interaction_matrix, dtype=float)
    n = len(interaction_matrix)

    if interaction_matrix.shape != (n, n):
        raise ValueError('interaction_matrix must be square')
    if set(G_qpu.nodes) != set(range(n)):
        raise ValueError('G_qpu nodes must be 0..{} to match interaction_matrix'.format(n - 1))

    # fixed node order: column k of the adjacency matrix is node k, whatever the insertion order in G_qpu
    adjacency = nx.to_numpy_array(G_qpu, nodelist=range(n))

    # the swap gains assume symmetric matrices (for a symmetric interaction_matrix this
    # leaves the score unchanged, also when G_qpu is a DiGraph)
    A = (interaction_matrix + interaction_matrix.T) / 2
    B = (adjacency + adjacency.T) / 2

    if n < 2:
        return np.arange(n), _score(interaction_matrix, adjacency, np.arange(n))

    rng = np.random.default_rng(seed)
    n_iter = 100 * n if n_iter is None else n_iter

    starts = [] if initial_guess is None else [np.asarray(initial_guess, dtype=int)]
    for _ in range(n_restarts):
        res_faq = quadratic_assignment(A, B, method='faq', options={'maximize': True, 'P0': 'randomized', 'rng': rng})
        starts.append(res_faq.col_ind)

    solutions = [_tabu_search(A, B, p0, n_iter, rng) for p0 in starts]
    scores = [_score(A, B, p) for p in solutions]
    best = solutions[int(np.argmax(scores))]

    if verbose:
        n_best = int(np.sum(np.isclose(scores, max(scores))))
        print('optimize_mapping: best score {:.6f} reached in {}/{} runs'.format(max(scores), n_best, len(scores)))

    return best, _score(interaction_matrix, adjacency, best)


def print_permutation(G_qpu, permutation, centers=None, orbital_labels=True):
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

    if centers is None:
        pos = nx.spring_layout(G_qpu)
    else:
        pos = {permutation[i]: centers[i][:2] for i in range(n_orbitals)}

    permutation_inv = np.argsort(permutation).tolist()

    edge_labels = {}
    if pos is None:
        for i, j in G_qpu.edges:
            i2 = permutation_inv[i]
            j2 = permutation_inv[j]

            d = np.linalg.norm(centers[i2][:2] - centers[j2][:2])
            edge_labels[(i, j)] = f"{d:.2f}"


    if orbital_labels:
        labels = {i: permutation_inv[i] for i in G_qpu.nodes}
    else:
        labels = {i: i for i in G_qpu.nodes}

    #labels = {i: '{}->{}'.format(permutation_inv[i], i) for i in G_qpu.nodes}

    #nx.draw(G_qpu, pos, with_labels=False)

    # Edges with arrows
    edges_directed = [
        (u, v) for u, v, data in G_qpu.edges(data=True)
        if data.get("spin_trajectory", True)
    ]

    # Edges without arrows
    edges_undirected = [
        (u, v) for u, v, data in G_qpu.edges(data=True)
        if not data.get("spin_trajectory", True)
    ]

    nx.draw_networkx_nodes(G_qpu, pos)

    nx.draw_networkx_edges(
        G_qpu, pos, edgelist=edges_directed,
        arrows=True
    )

    nx.draw_networkx_edges(
        G_qpu, pos, edgelist=edges_undirected,
        arrows=False,
        #edge_color='red',
        style='dashed',
    )

    nx.draw_networkx_labels(G_qpu, pos, labels=labels)
    nx.draw_networkx_edge_labels(G_qpu, pos, edge_labels=edge_labels)

    plt.axis('off')

    #nx.draw_networkx_edge_labels(G_qpu, pos, edge_labels=edge_labels)
    #nx.draw_networkx_labels(G_qpu, pos,labels=labels)
    plt.show()
