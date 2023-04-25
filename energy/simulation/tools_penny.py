from utils import convert_hamiltonian, string_to_matrix
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
import pennylane as qml
import numpy as np
import scipy


def build_gradient_ansatz(hf_reference_fock, matrix):

    # Initialize qubits
    n_qubits = len(hf_reference_fock)

    # dev1 = qml.device('default.qubit', wires=n_qubits)
    state_preparation_gates = [qml.PauliX(wires=[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    # Append the ansatz directly as a matrix
    state_preparation_gates.append(qml.QubitUnitary(matrix.toarray(), wires=list(range(n_qubits))))

    return state_preparation_gates


def get_preparation_gates(coefficients, ansatz, hf_reference_fock, n_qubits):

    # Create sparse 2x2 identity matrix and initialize the ansatz matrix with it
    identity = scipy.sparse.identity(2, format='csc', dtype=complex)

    # Multiply the ansatz matrix by identity as many times as necessary to get
    # the correct dimension
    matrix = identity
    for _ in range(n_qubits - 1):
        matrix = scipy.sparse.kron(identity, matrix, 'csc')

    # Multiply the identity matrix by the matrix form of each operator in the
    # ansatz, to obtain the matrix representing the action of the complete ansatz
    for coefficient, operator in zip(coefficients, ansatz):
        # Get corresponding the sparse operator, with the correct dimension
        # (forcing n_qubits = qubitNumber, even if this operator acts on less
        # qubits)
        operator_matrix = get_sparse_operator(coefficient * operator, n_qubits)

        # Multiply previous matrix by this operator
        matrix = scipy.sparse.linalg.expm(operator_matrix) * matrix

    # Prepare ansatz directly as a matrix f
    state_preparation_gates = build_gradient_ansatz(hf_reference_fock, matrix)

    return state_preparation_gates


def measure_expectation(main_string, sub_hamiltonian, shots, state_preparation_gates, n_qubits):
    """
    Measures the expectation value of a subHamiltonian using the CIRQ simulator
    (simulating sampling). By construction, all the expectation values of the
    strings in subHamiltonian can be obtained from the same measurement array.

    Arguments:
      main_string (str): the main Pauli string. This is the string in the group
        with the least identity terms. It defines the circuit that will be used.
      sub_hamiltonian (dict): a dictionary whose keys are boolean strings
        representing substrings of the main one, and whose values are the
        respective coefficients.
      shots (int): the number of repetitions to be performed, the
      state_preparation_gates (list): the list of CIRQ gates that prepare (from
        |0..0>) the state in which to obtain the expectation value.
      qubits (list): list of cirq.LineQubit to apply the gates on

    Returns:
      total_expectation_value (float): the measurements expectation value of
        subHamiltonian, with sampling noise.
    """

    # Initialize circuit.
    dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)], shots=shots)

    # add gates to circuit
    def circuit_function():
        for gate in state_preparation_gates:
            qml.apply(gate)

        for i, op in enumerate(main_string):
            if op == "X":
                qml.Hadamard(wires=[i])

            # Rotate qubit i to the Y basis if that's the desired measurement.
            elif op == "Y":
                qml.RX(np.pi / 2, wires=[i])
        return qml.sample()
        # return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # Append to the circuit the gates that prepare the state corresponding to
    # the received parameters.
    circuit = qml.QNode(circuit_function, dev_unique_wires, analytic=None)
    # print(qml.draw(circuit)())

    result_data = {}
    if main_string != "I" * n_qubits:
        raw_results = np.array(circuit()).T
        for i, measure_vector in enumerate(raw_results):
            result_data.update({'{}'.format(i): measure_vector.tolist()})
    else:
        raise Exception('Nothing to run')

    # For each substring, initialize the sum of all measurements as zero
    measurements = {}
    for sub_string in sub_hamiltonian:
        measurements[sub_string] = 0

    for vector in raw_results.T:
        for sub_string in sub_hamiltonian:

            prod_function = 1
            for i, v in enumerate(vector):
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
    '''
    Calculates the exact energy in a specific state using matrix algebra

    Arguments:
      state_vector (np.ndarray): the state in which to obtain the
        expectation value.
      qubit_hamiltonian (dict): the Hamiltonian of the system.

    Returns:
      exact_evaluation (float): the expectation value in the state given the hamiltonian.
    '''

    n_qubits = count_qubits(qubit_hamiltonian)

    # Initialize circuit.
    dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)])

    # add gates to circuit
    def circuit_function():
        for gate in state_preparation_gates:
            qml.apply(gate)
        return qml.state()

    # create and run circuit
    circuit = qml.QNode(circuit_function, dev_unique_wires, analytic=None)
    state_vector = circuit()

    formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)

    # Obtain the theoretical expectation value for each Pauli string in the
    # Hamiltonian by matrix multiplication, and perform the necessary weighed
    # sum to obtain the energy expectation value.
    exact_evaluation = 0
    for pauli_string in formatted_hamiltonian:
        ket = np.array(state_vector, dtype=complex)
        bra = np.conj(ket)

        pauli_ket = np.matmul(string_to_matrix(pauli_string), ket)
        expectation_value = np.real(np.dot(bra, pauli_ket))

        exact_evaluation += formatted_hamiltonian[pauli_string] * expectation_value

    return exact_evaluation.real
