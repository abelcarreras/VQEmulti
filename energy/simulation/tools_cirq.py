from utils import convert_hamiltonian, string_to_matrix
from openfermion import get_sparse_operator
import numpy as np
import scipy
import cirq


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
      total_expectation_value (float): the total expectation value of
        subHamiltonian, with sampling noise.
    """

    # Initialize circuit.
    circuit = cirq.Circuit()

    # Append to the circuit the gates that prepare the state corresponding to
    # the received parameters.
    circuit.append(state_preparation_gates)

    # cirq.optimizers.EjectZ().optimize_circuit(circuit)
    # cirq.optimizers.DropNegligible().optimize_circuit(circuit)

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
        s = cirq.Simulator()
        results = s.run(circuit, repetitions=shots)
    else:
        raise Exception('Nothing to run')

    # For each substring, initialize the sum of all measurements as zero
    measurements = {}
    for sub_string in sub_hamiltonian:
        measurements[sub_string] = 0

    # Calculate the expectation value of each Pauli string by averaging over
    # all the repetitions
    for j in range(shots):
        meas = {}

        # Initialize the measurement in repetition j for all substrings
        for sub_string in sub_hamiltonian:
            meas[sub_string] = 1

        # Go through the measurements on all the qubits
        for i in range(n_qubits):

            if main_string[i] != "I":
                # There's a measurement associated with this qubit

                # Use this single qubit measurement for the calculation of the
                # measurement of each full substring in this repetition. If the
                # substring has a "0" in the position corresponding to this
                # qubit, the operator associated is I, and the measurement
                # is ignored (raised to the power of 0)
                for sub_string in sub_hamiltonian:
                    #print(results.data[str(i)][j], (1 - 2 * results.data[str(i)][j]), sub_string[i])
                    meas[sub_string] = meas[sub_string] * ((1 - 2 * results.data[str(i)][j]) ** int(sub_string[i]))

        # Add this measurement to the total, for each string
        for sub_string in sub_hamiltonian:
            measurements[sub_string] += meas[sub_string]

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

    circuit = cirq.Circuit(state_preparation_gates)
    s = cirq.Simulator()

    # Access the exact final state vector
    results = s.simulate(circuit)
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
    statePreparationGates = [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    # Append the ansatz directly as a matrix
    statePreparationGates.append(cirq.MatrixGate(matrix.toarray()).on(*qubits))

    return statePreparationGates


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


def get_circuit_depth(state_preparation_gates):
    circuit = cirq.Circuit(state_preparation_gates)
    return len(cirq.Circuit(circuit.all_operations()))

