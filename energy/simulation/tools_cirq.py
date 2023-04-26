from utils import convert_hamiltonian, string_to_matrix
from openfermion import get_sparse_operator
import numpy as np
import scipy
import cirq


def measure_expectation(main_string, sub_hamiltonian, shots, state_preparation_gates, n_qubits):
    """
    Measures the expectation value of a sub_hamiltonian using the Cirq simulator.
    By construction, all the expectation values of the strings in subHamiltonian can be
    obtained from the same measurement array. This reduces quantum computer simulations

    :param main_string: hamiltonian main string ex: (XXYY)
    :param sub_hamiltonian: partial hamiltonian interactions ex: {'0000': -0.4114, '1111': -0.0222}
    :param shots: number of samples to simulate
    :param state_preparation_gates: list of gates in simulation library format that represents the state
    :param n_qubits: number of qubits
    :return:
    """

    # Initialize circuit.
    circuit = cirq.Circuit()

    # Append to the circuit the gates that prepare the state corresponding to
    # the received parameters.
    circuit.append(state_preparation_gates)

    # optimize circuit
    circuit = cirq.eject_z(circuit)
    circuit = cirq.drop_negligible_operations(circuit)

    # define qubits
    qubits = cirq.LineQubit.range(n_qubits)

    # parse string
    # Append necessary rotations and measurements for each qubit.
    for i, qubit in enumerate(qubits):
        op = main_string[i]

        # Rotate qubit i to the X basis if that's the desired measurement.
        if op == "X":
            circuit.append(cirq.H(qubit))

        # Rotate qubit i to the Y basis if that's the desired measurement.
        elif op == "Y":
            circuit.append(cirq.rx(np.pi / 2).on(qubit))

        # Measure qubit i in the computational basis, unless operator is I.
        if op != "I":
            circuit.append(cirq.measure(qubit, key=str(i)))

    # Sample the desired number of repetitions from the circuit, unless
    # there are no measurements (identity term).
    if main_string != "I" * n_qubits:
        simulation = cirq.Simulator()
        results = simulation.run(circuit, repetitions=shots)
    else:
        raise Exception('Nothing to run')

    # For each substring, initialize the sum of all measurements as zero
    measurements = {}
    for sub_string in sub_hamiltonian:
        measurements[sub_string] = 0

    indices = np.array(results.data.T.index, dtype=int)
    for vector in results.data.values:
        for sub_string in sub_hamiltonian:
            prod_function = 1
            for i, v in zip(indices, vector):
                if main_string[i] != "I":
                    prod_function *= int(1 - 2 * v) ** int(sub_string[i])

            measurements[sub_string] += prod_function

    # Calculate the expectation value of the subHamiltonian, by multiplying
    # the expectation value of each substring by the respective coefficient
    total_expectation_value = 0
    for sub_string, coefficient in sub_hamiltonian.items():
        # Get the expectation value of this substring by taking the average
        # over all the repetitions
        expectation_value = measurements[sub_string] / shots

        # Add this value to the measurements expectation value, weighed by its
        # coefficient
        total_expectation_value += expectation_value * coefficient

    return total_expectation_value


def get_exact_state_evaluation(qubit_hamiltonian, state_preparation_gates):
    """
    Calculates the exact energy in a specific state using matrix algebra.
    This function is basically used to test that the Cirq circuit is correct

    :param qubit_hamiltonian: hamiltonian in qubits
    :param state_preparation_gates: list of gates in simulation library format that represents the state
    :return: the expectation value of the state given the hamiltonian
    """

    circuit = cirq.Circuit(state_preparation_gates)
    simulation = cirq.Simulator()

    # Access the exact final state vector
    results = simulation.simulate(circuit)
    state_vector = results.final_state_vector

    formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)

    exact_evaluation = 0

    # Obtain the theoretical expectation value for each Pauli string in the
    # Hamiltonian by matrix multiplication, and perform the necessary weighed
    # sum to obtain the energy expectation value.
    for pauli_string in formatted_hamiltonian:
        ket = np.array(state_vector, dtype=complex)
        bra = np.conj(ket)

        pauli_ket = np.matmul(string_to_matrix(pauli_string), ket)
        expectation_value = np.real(np.dot(bra, pauli_ket))

        exact_evaluation += formatted_hamiltonian[pauli_string] * expectation_value

    return exact_evaluation.real


def build_reference_gates(hf_reference_fock):

    # Initialize qubits
    n_qubits = len(hf_reference_fock)
    qubits = cirq.LineQubit.range(n_qubits)

    # Create the gates for preparing the Hartree Fock ground state, that serves
    # as a reference state the ansatz will act on
    return [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]


def build_gradient_ansatz(hf_reference_fock, matrix):

    # Initialize qubits
    n_qubits = len(hf_reference_fock)
    qubits = cirq.LineQubit.range(n_qubits)

    # Initialize the state preparation gates with the Hartree Fock preparation
    state_preparation_gates = [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    # Append the ansatz directly as a matrix
    state_preparation_gates.append(cirq.MatrixGate(matrix.toarray()).on(*qubits))

    return state_preparation_gates

def get_circuit_depth(state_preparation_gates):
    circuit = cirq.Circuit(state_preparation_gates)
    return len(cirq.Circuit(circuit.all_operations()))

