from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
import numpy as np

import warnings

def get_n_particles_operator(n_orbitals, to_pauli=False):
    from openfermion import FermionOperator, jordan_wigner

    sz = FermionOperator()
    for i in range(0, n_orbitals * 2):
        a = FermionOperator((i, 1))
        a_dag = FermionOperator((i, 0))
        sz += a * a_dag

    if to_pauli:
        sz = jordan_wigner(sz)
    return sz

warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 2
config.mapping = 'jw'


from vqemulti.utils import load_wave_function, load_hamiltonian
coeff, ansatz = load_wave_function(filename='wf_qubit_trotter_6.yml', qubit_op=True)
# coeff, ansatz = load_wave_function(filename='wf_fermion.yml', qubit_op=False)

print(coeff)
print(ansatz)
#exit()


# set HF
#coeff = coeff[:0]
#ansatz = ansatz[:0]

hamiltonian = load_hamiltonian(file='hamiltonian.npz')
hamiltonian = get_fermion_operator(hamiltonian)
hamiltonian.compress(1e-2)
print('H terms', len(hamiltonian.terms))
n_electrons = 4
n_orbitals = 4

# hamiltonian = get_n_particles_operator(n_orbitals)


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)

pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)


from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeAlmadenV2, FakeKyiv, FakeProviderForBackendV2
from qiskit_aer import AerSimulator

backend = FakeTorino()
#backend = FakeAlmadenV2()
#backend = FakeVigoV2()
# backend = AerSimulator()

#print(backend.coupling_map)

# service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_torino')

print('backend: ', backend.name)

# Start session
print('Initialize Single point')



from qiskit import QuantumCircuit

def accumulated_errors(backend: QiskitRuntimeService.backend, circuit: QuantumCircuit) -> list:
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
    results = [
        acc_total_error,
        acc_two_qubit_error,
        acc_single_qubit_error,
        acc_readout_error,
        single_qubit_gate_count,
        two_qubit_gate_count,
    ]
    return results


def get_backend_opt_layout(backend, n_qubits, plot_data=False, cache_time=3600):
    from vqemulti.simulators.backend_opt import time, layouts_cache, get_paths

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


def get_backend_opt_layout_2(backend, n_qubits, plot_data=False, cache_time=3600):
    from vqemulti.simulators.backend_opt import time, layouts_cache, get_paths

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
    properties = backend.properties()

    # one_gate_error
    one_gate_error = np.ones((n_backend_qubits))

    acc_readout_error=0
    for q in range(n_backend_qubits):
        one_gate_error[q] = properties.readout_error(q)

    print(one_gate_error)

    # get total error
    two_gate_non_error = np.zeros((n_backend_qubits))
    two_gate_instructions = [op.name for op in backend.operations if op.num_qubits == 2 and isinstance(op.name, str)]

    for instruc in two_gate_instructions:
        for qubits, value in backend.target[instruc].items():
            print(qubits)
            for q in qubits:
                two_gate_non_error[q] += value.error/2


    print(two_gate_non_error)

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


with Session(backend=backend) as session:

    energies = []
    variances = []

    for i in range(30):

        # coeff_ = coeff[:i]
        # ansatz_ = ansatz[:i]

        simulator = Simulator(trotter=True,
                              trotter_steps=1,
                              test_only=False,
                              hamiltonian_grouping=True,
                              session=session,
                              use_estimator=True,
                              shots=9000)

        # run SP energy
        #simulator = None
        energy = get_adapt_vqe_energy(coeff,
                                      ansatz,
                                      hf_reference_fock,
                                      hamiltonian,
                                      simulator)

        print('energy final: ', energy)

        #print(simulator.get_circuits()[-1])
        #simulator.print_statistics()

        energies.append(energy)
        #simulator.print_statistics()
        #exit()


print('\nEnergy list')
for e in energies:
    print(e)

print('ave: ', np.average(energies))
# print('std: ', np.std(energies))
# print('var: ', np.var(energies))


