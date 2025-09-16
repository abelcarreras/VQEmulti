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


def get_backend_opt_layout(backend, n_qubits, plot_data=False, cache_time=3600):

    # layout cache
    time.time()
    current_time = time.time()
    if backend.name in layouts_cache:
        if abs(layouts_cache[backend.name]['time'] - current_time) < cache_time:
            return layouts_cache[backend.name]['layout']

    # check if backend belongs to IBM runtime
    if isinstance(backend, str):
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        backend = service.backend(backend)

    if backend.coupling_map is None:
        warnings.warn('Unable to generate backend layout')
        return None

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
        print('Quality layout: ', highest_quality)

        plt.show()


    # store layout
    layouts_cache[backend.name] = {'time': current_time, 'layout': layout}

    return layout


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


if __name__ == '__main__':

    # fake backend
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2, FakeTorino
    backend = FakeTorino()

    # real hardware backend
    backend = 'ibm_torino'

    layout = get_backend_opt_layout(backend, 4, plot_data=True)

    print('layout: ', layout)