from collections import defaultdict
import warnings
import numpy as np
import time

layouts_cache = {}


def create_graph(parelles):
    graf = defaultdict(list)
    for (u, v) in parelles:
        graf[u].append(v)
        graf[v].append(u)
    return graf


def find_path(graf, node_actual, N, cami_actual, tots_camins):
    if len(cami_actual) == N:
        tots_camins.append(list(cami_actual))
        return
    for vei in graf[node_actual]:
        if vei not in cami_actual:  # avoid cycles
            cami_actual.append(vei)
            find_path(graf, vei, N, cami_actual, tots_camins)
            cami_actual.pop()  # Desfer l'últim pas

# main function
def get_paths(parelles, N):
    graf = create_graph(parelles)
    tots_camins = []
    for node in graf:
        find_path(graf, node, N, [node], tots_camins)
    return tots_camins


# draw graph
def draw_graph(G,
               pos=None,
               ax=None,
               node_size=600,
               node_color="tab:blue",
               edge_color="black",
               with_labels=True,
               labels=None):
    """
    Draw a NetworkX graph.

    Parameters
    ----------
    G : nx.Graph
        Graph to draw.
    pos : dict, optional
        Dictionary mapping nodes to (x, y) coordinates.
        If None, spring_layout is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    if pos is None:
        pos = nx.spring_layout(G, seed=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_color,
        width=2,
        ax=ax,
    )

    if with_labels:
        if labels is None:
            labels = {n: str(n) for n in G.nodes}

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=10,
            ax=ax,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    plt.show()

    return pos


# score subgraph
def score_subgraph(backend, qubits,
                   w_readout=1.0,
                   w_1q=1.0,
                   w_2q=2.0,
                   w_t1=0.2,
                   w_t2=0.2
                   ):
    """
    Score a subset of physical qubits.

    Higher score is better.

    Parameters
    ----------
    backend : BackendV2
    qubits : iterable[int]
        Physical qubit indices.
    """

    import math
    import statistics

    target = backend.target
    qubits = list(qubits)
    qset = set(qubits)

    readout = []
    oneq = []
    t1 = []
    t2 = []
    twoq = []

    # ------------------------------
    # Single-qubit properties
    # ------------------------------

    for q in qubits:

        props = target.qubit_properties[q]

        if props is not None:
            if props.t1 is not None:
                t1.append(props.t1)

            if props.t2 is not None:
                t2.append(props.t2)

        # Readout error
        if "measure" in target:
            inst = target["measure"].get((q,))
            if inst is not None and inst.error is not None:
                readout.append(1.0 - inst.error)

        # Best single-qubit gate
        best = []

        for gate in ("sx", "x", "rz"):
            if gate in target:
                inst = target[gate].get((q,))
                if inst is not None and inst.error is not None:
                    best.append(1.0 - inst.error)

        if best:
            oneq.append(max(best))

    # ------------------------------
    # Two-qubit fidelities
    # ------------------------------

    for gate in ("cx", "ecr", "cz"):

        if gate not in target:
            continue

        for edge, inst in target[gate].items():

            if edge[0] in qset and edge[1] in qset:

                if inst.error is not None:
                    twoq.append(1.0 - inst.error)

    # ------------------------------
    # Aggregate
    # ------------------------------

    def mean_or_zero(values):
        return statistics.mean(values) if values else 0.0

    score = 0.0

    score += w_readout * mean_or_zero(readout)
    score += w_1q * mean_or_zero(oneq)
    score += w_2q * mean_or_zero(twoq)

    if t1:
        score += w_t1 * math.log10(mean_or_zero(t1))

    if t2:
        score += w_t2 * math.log10(mean_or_zero(t2))

    return score



class LayoutModelDefault:
    def __init__(self, custom_list=None):
        self._qubits_list = custom_list

    def get_layout(self, circuit, backend, n_qubits):
        if self._qubits_list is not None and len(self._qubits_list) < n_qubits and backend.num_qubits > n_qubits:
            raise Exception('Number of requested qubits ({}) is larger than available'.format(n_qubits))
        return self._qubits_list

    def plot_data(self, backend, n_qubits):
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        n_backend_qubits = backend.num_qubits

        layout = self.get_layout(None, backend, n_qubits)

        if layout is not None:
            qubit_color = []
            for i in range(n_backend_qubits):
                if i in layout:
                    qubit_color.append("#ff0066")
                else:
                    qubit_color.append("#6600cc")
        else:
            qubit_color = None

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()

class LayoutModelLinear:
    def __init__(self, cache_time=3600):
        self._cache_time = cache_time

    def _process_data(self, backend):

        # get data
        n_backend_qubits = backend.num_qubits

        decoherence_t1 = []
        decoherence_t2 = []
        for i in range(n_backend_qubits):
            decoherence_t1.append(backend.qubit_properties(i).t1)
            decoherence_t2.append(backend.qubit_properties(i).t2)

        # get total error
        two_gate_non_error = np.ones((n_backend_qubits))
        two_gate_instructions = [op.name for op in backend.operations if op.num_qubits == 2 and isinstance(op.name, str)]
        # print('instructions:', two_gate_instructions)
        # print('operations: ', backend.operations)

        for instruc in two_gate_instructions:
            #print(backend.target[instruc])
            for qubits, value in backend.target[instruc].items():
                for q in qubits:
                    two_gate_non_error[q] *= (1.0 - value.error)

        # print(two_gate_non_error)

        quality = two_gate_non_error/np.average(two_gate_non_error) + \
                  np.array(decoherence_t1)/np.average(decoherence_t1) + \
                  np.array(decoherence_t2)/np.average(decoherence_t2)

        return quality, two_gate_non_error, decoherence_t1, decoherence_t2

    def get_layout(self, circuit, backend, n_qubits):

        # check if backend belongs to IBM runtime
        if isinstance(backend, str):
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService()
            backend = service.backend(backend)

        if backend.coupling_map is None:
            warnings.warn('Unable to generate backend layout with {}'.format(backend))
            return None

        # layout cache
        time.time()
        current_time = time.time()
        if (backend.name, n_qubits) in layouts_cache:
            if abs(layouts_cache[(backend.name, n_qubits)]['time'] - current_time) < self._cache_time:
                return layouts_cache[(backend.name, n_qubits)]['layout']

        quality, two_gate_non_error, decoherence_t1, decoherence_t2 = self._process_data(backend)

        edges = backend.coupling_map.get_edges()

        # find optimal path
        layout = None
        self._highest_quality = 0
        for path in get_paths(edges, n_qubits):
            quality_path = sum([quality[j] for j in path])
            if quality_path > self._highest_quality:
                self._highest_quality = quality_path
                layout = path

        # store layout
        layouts_cache[(backend.name, n_qubits)] = {'time': current_time, 'layout': layout}

        return layout

    def plot_data(self, backend, n_qubits):
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        n_backend_qubits = backend.num_qubits

        quality, two_gate_non_error, decoherence_t1, decoherence_t2 = self._process_data(backend)

        plt.title('Error per qubits (lower is better)')
        plt.xlabel('Qubits')
        plt.ylabel('Error')
        plt.bar([str(i) for i in range(n_backend_qubits)], 1 - two_gate_non_error)

        plt.figure()
        plt.title('Decoherence time T1 (higher is better)')
        plt.xlabel('Qubits')
        plt.ylabel('time (s)')
        plt.bar([str(i) for i in range(n_backend_qubits)], decoherence_t1)

        plt.figure()
        plt.title('Decoherence time T2 (higher is better)')
        plt.xlabel('Qubits')
        plt.ylabel('time (s)')
        plt.bar([str(i) for i in range(n_backend_qubits)], decoherence_t2)

        layout = self.get_layout(None, backend, n_qubits)
        qubit_color = []
        for i in range(n_backend_qubits):
            if i in layout:
                qubit_color.append("#ff0066")
            else:
                qubit_color.append("#6600cc")

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        print('Quality layout: ', self._highest_quality)

        plt.show()


class LayoutModelSQD:
    def __init__(self, initial_qubit=21, cache_time=3600):
        self._cache_time = cache_time
        self._initial_qubit = initial_qubit

    def get_layout(self, circuit, backend, n_qubits):

        import networkx as nx

        # check if backend belongs to IBM runtime
        if isinstance(backend, str):
            from qiskit_ibm_runtime import QiskitRuntimeService

            service = QiskitRuntimeService()
            backend = service.backend(backend)

        if backend.coupling_map is None:
            warnings.warn('Unable to generate backend layout with {}'.format(backend))
            return None

        def build_coupling_graph(backend):
            """
            Convert the backend coupling map to a NetworkX undirected graph.
            """
            rw_graph = backend.coupling_map.graph

            # convert to networkx
            G = nx.Graph()
            G.add_nodes_from(rw_graph.nodes())
            for edge in rw_graph.edge_list():
                G.add_edge(edge[0], edge[1])

            return G

        # build Xnetwork from compling map
        G = build_coupling_graph(backend)

        # generate graph from centers of heavy hex
        nodes_degree_3 = [n for n, d in G.degree() if d == 3]
        G_big = nx.Graph()
        for i, i_node in enumerate(nodes_degree_3):
            idx = nodes_degree_3.index(i_node)
            G_big.add_node(idx)

        for i, i_node in enumerate(nodes_degree_3):
            for j_node in nodes_degree_3[i:]:
                d = nx.shortest_path_length(G, i_node, j_node)
                if d == 2:
                    # print(i_node, j_node)
                    idx_i = nodes_degree_3.index(i_node)
                    idx_j = nodes_degree_3.index(j_node)

                    G_big.add_edge(idx_i, idx_j)

        import matplotlib.pyplot as plt
        # pos = nx.kamada_kawai_layout(G_big)
        # nx.draw(G_big, pos, with_labels=True)
        # plt.show()
        # print(G_big)

        # nodes with a single edge (border nodes)
        nodes_degree_1 = [n for n, d in G_big.degree() if d == 1]

        # compute distance vector
        dist_vec = []
        nodes_list = []
        for i in nodes_degree_1:
            for j in nodes_degree_1:
                if i != j:
                    d = nx.shortest_path_length(G_big, i, j)
                    dist_vec.append(d)
                    nodes_list.append((i, j))

        # compute compatibility between paths (not crossing)
        compatible = np.zeros((len(nodes_list), len(nodes_list)), dtype=bool)
        for i, i_pair in enumerate(nodes_list):
            for j, j_pair in enumerate(nodes_list):
                path_i = nx.shortest_path(G_big, source=i_pair[0], target=i_pair[1])
                path_j = nx.shortest_path(G_big, source=j_pair[0], target=j_pair[1])
                # print('->', i, j, not np.any([i in path_j for i in path_i]), [i in path_j for i in path_i])
                compatible[i, j] = not np.any([i in path_j for i in path_i])

        # sort by distance vector
        arg_sort = np.argsort(dist_vec)[::-1]
        dist_vec = np.array(dist_vec)[arg_sort]
        # print('dist_vec: ', dist_vec)
        compatible = compatible[:, arg_sort][arg_sort, :]
        nodes_list = np.array(nodes_list)[arg_sort]

        sum_vector = []
        pair = []
        for i in arg_sort:
            for j in arg_sort:
                if (nodes_list[i][0] not in nodes_list[j]) and (nodes_list[i][1] not in nodes_list[j]):
                    if compatible[i, j]:
                        #pair_beta = nodes_list[i]
                        sum_vector.append(dist_vec[i] + dist_vec[j])
                        pair.append([nodes_list[i], nodes_list[j]])

        max_index = np.argsort(sum_vector)[-1]
        pair_alpha, pair_beta = pair[max_index]
        #print(pair_alpha)
        #print(pair_beta)

        path_a = nx.shortest_path(G_big, source=pair_alpha[0], target=pair_alpha[1])
        path_b = nx.shortest_path(G_big, source=pair_beta[0], target=pair_beta[1])
        assert not np.any([i in path_b for i in path_a])


        def big_to_original(G, path_i, nodes_degree_3):
            path = []
            for i in range(len(path_i)-1):
                ini = nodes_degree_3[path_i[i]]
                fin = nodes_degree_3[path_i[i+1]]
                path += nx.shortest_path(G, source=ini, target=fin)[:-1]
            return path

        alpha_chain = big_to_original(G, path_a, nodes_degree_3)
        beta_chain = big_to_original(G, path_b, nodes_degree_3)

        n_nodes = n_qubits//2
        alpha_chain = alpha_chain[-n_nodes:]
        beta_chain = beta_chain[:n_nodes]

        layout = []
        for qa, qb in zip(alpha_chain, beta_chain):
            layout += [qa, qb]

        # print('layout :', layout)
        return layout

    def plot_data(self, backend, n_qubits):
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        n_backend_qubits = backend.num_qubits

        layout = self.get_layout(None, backend, n_qubits)
        qubit_color = []
        for i in range(n_backend_qubits):
            if i in layout:
                qubit_color.append("#ff0066")
            else:
                qubit_color.append("#6600cc")

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()


def accumulated_errors(backend, circuit, print_data=False):
    """Compute accumulated gate and readout errors for a given circuit on a specific backend."""

    # Initializing quantities
    acc_single_qubit_error = 0
    acc_two_qubit_error = 0
    single_qubit_gate_count = 0
    two_qubit_gate_count = 0
    acc_readout_error = 0

        # Defining useful variables
    properties = backend.properties()
    if properties is None:
        warnings.warn('Unable to compute accumulated errors with this backend')
        return
    qubit_layout = list(circuit.layout.initial_layout.get_physical_bits().keys())#[:n]

    # Define readout error (only for qubits in qubit_layout) using `properties.readout_error`
    for q in qubit_layout:
        acc_readout_error+= properties.readout_error(q)

    # Define two qubit gates for the different backends using `backend.configuration()`
    config = backend.configuration()
    if "ecr" in config.basis_gates:
        two_qubit_gate = "ecr"
    elif "cz" in config.basis_gates:
        two_qubit_gate = "cz"
    # Loop over the instructions in `circuit.data` to account for the single and two-qubit errors and single and two qubit gate counts
    for instruction, qargs, _ in circuit.data:
        # print(instruction)
        qubit_indices = [circuit.find_bit(q).index for q in qargs]
        # print(qubit_indices)

        if instruction.name == 'measure':
            continue
        if instruction.num_qubits == 1: # Count and add errors for one qubit gates
            acc_single_qubit_error += properties.gate_error(instruction.name, qubit_indices)
            single_qubit_gate_count += 1
        elif instruction.num_qubits == 2: # Count and add errors for two qubit gates
            acc_two_qubit_error += properties.gate_error(instruction.name, qubit_indices)
            two_qubit_gate_count += 1

    acc_total_error = acc_two_qubit_error + acc_single_qubit_error + acc_readout_error

    if print_data:
        print(f'Backend {backend.name}')
        print(f'Accumulated two-qubit error of {two_qubit_gate_count} gates: {acc_two_qubit_error:.3f}')
        print(f'Accumulated one-qubit error of {single_qubit_gate_count} gates: {acc_single_qubit_error:.3f}')
        print(f'Accumulated readout error: {acc_readout_error:.3f}')
        print(f'Accumulated total error: {acc_total_error:.3f}\n')

    results = [
        acc_total_error,
        acc_two_qubit_error,
        acc_single_qubit_error,
        acc_readout_error,
        single_qubit_gate_count,
        two_qubit_gate_count,
    ]
    return results


class LayoutModelNapo:
    def __init__(self, custom_list=None):
        self._qubits_list = custom_list

    def get_layout(self, circuit, backend, n_qubits):

        import mapomatic as mm

        from qiskit import transpile
        trans_qc = transpile(circuit, backend)
        small_qc = mm.deflate_circuit(trans_qc)

        layouts = mm.matching_layouts(small_qc, backend)

        #layouts = layouts[:2]
        #print(layouts)
        #print('num qubits: ', small_qc.num_qubits)

        scores = mm.evaluate_layouts(small_qc, layouts, backend)
        return scores[0][0]

    def plot_data(self, backend, n_qubits, circuit):
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        n_backend_qubits = backend.num_qubits

        layout = self.get_layout(circuit, backend, n_qubits)

        if layout is not None:
            qubit_color = []
            for i in range(n_backend_qubits):
                if i in layout:
                    qubit_color.append("#ff0066")
                else:
                    qubit_color.append("#6600cc")
        else:
            qubit_color = None

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()


class LayoutModelGraph:
    def __init__(self, G_orbital):

        import networkx as nx

        G_spin = nx.Graph()

        # Add alpha-beta pairs
        for i in G_orbital.nodes:
            alpha = 2 * i
            beta = alpha + 1

            G_spin.add_node(alpha)
            G_spin.add_node(beta)

        # Connect neighbouring orbitals
        for i, j in G_orbital.edges:
            alpha_i = 2 * i
            beta_i = alpha_i + 1
            alpha_j = 2 * j
            beta_j = alpha_j + 1

            # print(alpha_i, beta_i, alpha_j)

            G_spin.add_edge(alpha_i, beta_i)
            G_spin.add_edge(beta_i, alpha_j)
            G_spin.add_edge(alpha_j, beta_j)

        self._G_circuit = G_spin

        # draw_graph(G_spin)


    def get_layout(self, circuit, backend, n_qubits):

        from networkx.algorithms import isomorphism
        import networkx as nx

        # print(backend.coupling_map)
        # for i in backend.coupling_map:
        #    print(i)

        if backend.coupling_map is None:
            raise Exception('Unable to generate backend layout with {}'.format(backend))

        G_backend = nx.Graph()
        G_backend.add_edges_from(backend.coupling_map)

        # Find one embedding
        GM = isomorphism.GraphMatcher(G_backend, self._G_circuit)
        # mapping = next(GM.subgraph_isomorphisms_iter())

        unique_subgraphs = set()
        for mapping in GM.subgraph_isomorphisms_iter():
            nodes = frozenset(mapping.keys())  # set of nodes of G
            unique_subgraphs.add(nodes)

        scores = {}
        for subgraph in unique_subgraphs:
            score = score_subgraph(backend, subgraph)
            scores[subgraph] = score

            # print('score: ', score, ':', list(subgraph))

            plot_map = False
            if plot_map:
                import matplotlib.pyplot as plt

                # Edges of the subgraph inside G
                edges_subgraph = [(u, v) for u, v in G_backend.edges()
                                  if u in subgraph and v in subgraph]

                # Layout
                pos = nx.spring_layout(G_backend, seed=42)

                plt.figure()

                # Draw full graph (background)
                nx.draw(G_backend, pos,
                        with_labels=True,
                        node_color='lightgray',
                        edge_color='lightgray')

                # Highlight nodes of H
                nx.draw_networkx_nodes(G_backend, pos,
                                       nodelist=subgraph,
                                       node_color='red')

                # Highlight edges of H
                nx.draw_networkx_edges(G_backend, pos,
                                       edgelist=edges_subgraph,
                                       edge_color='red',
                                       width=2)

                plt.title("Subgraph H highlighted inside G")
                plt.show()

        return max(scores, key=scores.get)

    def plot_data(self, backend, n_qubits, circuit):
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        layout = self.get_layout(circuit, backend, n_qubits)

        n_backend_qubits = backend.num_qubits

        if layout is not None:
            qubit_color = []
            for i in range(n_backend_qubits):
                if i in layout:
                    qubit_color.append("#ff0066")
                else:
                    qubit_color.append("#6600cc")
        else:
            qubit_color = None

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()


if __name__ == '__main__':

    # fake backend
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2, FakeTorino
    backend = FakeTorino()

    # real hardware backend
    backend = 'ibm_torino'

    print('layout: ', layout)