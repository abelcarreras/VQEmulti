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
    # print('layout check')

    if isinstance(backend, str):
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backend = service.backend(backend)

    # get data
    n_backend_qubits = backend.num_qubits
    edges = backend.coupling_map.get_edges()

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

    # find optimal path
    layout = None
    highest_quality = 0
    for path in get_paths(edges, n_qubits):
        quality_path = sum([quality[j] for j in path])
        if quality_path > highest_quality:
            highest_quality = quality_path
            layout = path

    if plot_data:
        from qiskit.visualization import plot_gate_map
        import matplotlib.pyplot as plt

        plt.title('Error per qubits (lower is better)')
        plt.xlabel('Qubits')
        plt.ylabel('Error')
        plt.bar([str(i) for i in range(n_backend_qubits)], 1-two_gate_non_error)

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

        qubit_color = []
        for i in range(n_backend_qubits):
            if i in layout:
                qubit_color.append("#ff0066")
            else:
                qubit_color.append("#6600cc")

        plot_gate_map(backend, qubit_color=qubit_color, qubit_size=60, font_size=25, figsize=(8, 8))

        plt.show()

        print('Quality layout: ', highest_quality)

    return layout

if __name__ == '__main__':

    # fake backend
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2, FakeTorino
    backend = FakeTorino()

    # real hardware backend
    backend = 'ibm_torino'

    layout = get_backend_opt_layout(backend, 4, plot_data=True)

    print('layout: ', layout)