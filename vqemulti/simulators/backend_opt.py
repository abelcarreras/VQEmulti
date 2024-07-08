from collections import defaultdict
import numpy as np


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


def get_backend_opt_layout(backend, n_qubits, plot_data=False):
    print('layout check')

    if isinstance(backend, str):
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backend = service.backend(backend)

    # get data
    n_backend_qubits = backend.num_qubits
    edges = backend.coupling_map.get_edges()

    # get total error
    non_error = np.ones((n_backend_qubits))
    for target, data in backend.target.items():
        # print(target, data)
        #if target != 'measure':
        #    continue
        for qubits, value in data.items():
            if qubits is None:
                break
            for q in qubits:
                if value is None or value.error is None:
                    break
                non_error[q] *= (1.0 - value.error)

    # find optimal path
    layout = None
    highest_non_error = 0
    for path in get_paths(edges, n_qubits):
        non_error_path = sum([non_error[j] for j in path])
        # print(cami, error)
        if non_error_path > highest_non_error:
            highest_non_error = non_error_path
            layout = path

    if plot_data:
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        plt.figure()
        plt.title('Error per qubits')
        plt.xlabel('Qubits')
        plt.ylabel('Error')

        plt.bar([str(i) for i in range(n_backend_qubits)], 1-non_error)

        qubit_color = []
        for i in range(n_backend_qubits):
            if i in layout:
                qubit_color.append("#ff0066")
            else:
                qubit_color.append("#6600cc")

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()

        print('Average error layout: ', 1-highest_non_error)

    return layout

if __name__ == '__main__':

    # fake backend
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2, FakeTorino
    backend = FakeTorino()

    # real hardware backend
    backend = 'ibm_torino'

    layout = get_backend_opt_layout(backend, 4, plot_data=True)

    print('layout: ', layout)