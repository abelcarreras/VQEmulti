from utils import convert_hamiltonian, group_hamiltonian
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import scipy

# comment and uncomment to chage simulator
#from energy.simulation.trotter_cirq import get_preparation_gates_trotter, trotterizeOperator
#from energy.simulation.tools_cirq import measure_expectation, get_exact_state_evaluation, build_gradient_ansatz
from energy.simulation.trotter_penny import get_preparation_gates_trotter, trotterizeOperator
from energy.simulation.tools_penny import measure_expectation, get_exact_state_evaluation, build_gradient_ansatz


def get_preparation_gates(coefficients, ansatz, hf_reference_fock):
    """
    generate operation gates for a given ansantz in simulation library format (Cirq, pennylane, etc..)

    :param coefficients: operator coefficients
    :param ansatz: operators list in qubit
    :param hf_reference_fock: reference HF in fock vspace vector
    :param n_qubits: number of qubits
    :return: gates list in simulation library format
    """

    # generate matrix operator that corresponds to ansatz
    identity = scipy.sparse.identity(2, format='csc', dtype=complex)
    matrix = identity
    n_qubits = len(hf_reference_fock)
    for _ in range(n_qubits - 1):
        matrix = scipy.sparse.kron(identity, matrix, 'csc')

    for coefficient, operator in zip(coefficients, ansatz):
        # Get corresponding the operator matrix (exponent)
        operator_matrix = get_sparse_operator(coefficient * operator, n_qubits)

        # Add unitary operator to matrix as exp(operator_matrix)
        matrix = scipy.sparse.linalg.expm(operator_matrix) * matrix

    # Get gates in simulation library format
    state_preparation_gates = build_gradient_ansatz(hf_reference_fock, matrix)

    return state_preparation_gates


def get_sampled_energy(qubit_hamiltonian, shots, state_preparation_gates):
    """
    Obtains the expectation value in a state by sampling (using a simulator)

    :param qubit_hamiltonian: hamiltonian in qubits
    :param shots: number of samples
    :param state_preparation_gates: list of gates in simulation library format that represents the state
    :return: the expectation value of the energy
    """

    n_qubits = count_qubits(qubit_hamiltonian)

    # Format and group the Hamiltonian, so as to save measurements by using
    # the same data for Pauli strings that only differ by identities
    formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)
    grouped_hamiltonian = group_hamiltonian(formatted_hamiltonian)

    # Obtain the expectation value for each Pauli string by
    # calling the measure_expectation function, and perform the necessary weighed
    # sum to obtain the energy expectation value
    energy = 0
    for main_string, sub_hamiltonian in grouped_hamiltonian.items():
        expectation_value = measure_expectation(main_string,
                                                sub_hamiltonian,
                                                shots,
                                                state_preparation_gates,
                                                n_qubits)
        energy += expectation_value

    assert energy.imag < 1e-5

    return energy.real


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, qubit_hamiltonian, shots,
                        trotter=True, trotter_steps=1, test_only=True):
    """
    Obtain the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state, using the CIRQ simulator.

    :param coefficients: adaptVQE coefficients
    :param ansatz: list of qubit operators defining the current ansatz
    :param hf_reference_fock: reference HF in fock vspace vector
    :param qubit_hamiltonian: hamiltonian in qubits
    :param shots: number of samples
    :param trotter: if True trotterize ansatz operators (view Trotterization in QM Theory)
    :param trotter_steps: number of trotter steps
    :param test_only: if True evaluate circuit exactly, not sampling (for testing the circuit)
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    if trotter:
        #print('before trotter', ansatz)
        state_preparation_gates = get_preparation_gates_trotter(coefficients,
                                                                ansatz,
                                                                trotter_steps,
                                                                hf_reference_fock)

    else:
        state_preparation_gates = get_preparation_gates(coefficients,
                                                        ansatz,
                                                        hf_reference_fock)

    # from energy.simulation.tools import get_circuit_depth
    # circuit_depth = get_circuit_depth(state_preparation_gates)
    # print('circuit_depth', circuit_depth)

    if test_only:
        # Calculate the exact energy in this state (test circuit)
        energy = get_exact_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    else:
        # Obtain the energy expectation value by sampling from the circuit using the simulator
        energy = get_sampled_energy(qubit_hamiltonian, shots, state_preparation_gates)

    return energy
