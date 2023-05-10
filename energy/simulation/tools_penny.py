from utils import convert_hamiltonian, string_to_matrix
from openfermion import get_sparse_operator
from openfermion.utils import count_qubits
import pennylane as qml
import numpy as np
import scipy
import math
import matplotlib.pyplot

def build_gradient_ansatz(hf_reference_fock, matrix):

    # Initialize qubits
    n_qubits = len(hf_reference_fock)

    # Add gates for HF reference
    state_preparation_gates = [qml.PauliX(wires=[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    # Append the ansatz directly as a matrix
    state_preparation_gates.append(qml.QubitUnitary(matrix.toarray(), wires=list(range(n_qubits))))

    return state_preparation_gates

def build_uccsd_circuit_Nonia(coefficients,ansatz_dict,ansatz, hf_reference_fock):
    # Add gates for HF reference
    state_preparation_gates = [qml.PauliX(wires=[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]
    j=0


    for item in ansatz_dict.items():
        for i in item[0]:
            if i[1] == 'X':
                state_preparation_gates.append(qml.Hadamard(wires=i[0]))
            if i[1] == 'Y':
                state_preparation_gates.append(qml.RX(-math.pi/2, wires=i[0]))


        for i in range(3):
            state_preparation_gates.append(qml.CNOT(wires=[i, i+1]))

        state_preparation_gates.append(qml.RZ(coefficients[j],wires=3))


        for i in range(3,0,-1):
            state_preparation_gates.append(qml.CNOT(wires=[i-1, i]))

        for i in item[0]:
            if i[1] == 'X':
                state_preparation_gates.append(qml.Hadamard(wires=i[0]))
            if i[1] == 'Y':
                state_preparation_gates.append(qml.RX(math.pi / 2, wires=i[0]))
        j = j+1


    return state_preparation_gates


def get_preparation_gates(coefficients, ansatz, hf_reference_fock):
    """
    generate operation gates for a given ansantz in simulation library format (Cirq, pennylane, etc..)

    :param coefficients: operator coefficients
    :param ansatz: operators list in qubit
    :param hf_reference_fock: reference HF in fock vspace vector
    :return: gates list in simulation library format
    """

    n_qubits = len(hf_reference_fock)
    # generate matrix operator that corresponds to ansatz
    identity = scipy.sparse.identity(2, format='csc', dtype=complex)
    matrix = identity
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


def measure_expectation(main_string, sub_hamiltonian, shots, state_preparation_gates, n_qubits):
    """
    Measures the expectation value of a sub_hamiltonian using the Pennylane simulator.
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
    dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)], shots=shots)

    # Build circuit from preparation gates
    @qml.qnode(dev_unique_wires)
    def circuit():
        # apply preparation gates
        for gate in state_preparation_gates:
            qml.apply(gate)

        # apply hamiltonian gates according to main string
        for i, op in enumerate(main_string):
            if op == "X":
                qml.Hadamard(wires=[i])

            elif op == "Y":
                qml.RX(np.pi / 2, wires=[i])

        # sample measurements in PauliZ
        return [qml.sample(qml.PauliZ(wires=k)) for k in range(n_qubits)]

    # draw circuit
    # print(qml.draw(circuit)())

    result_data = {}
    if main_string != "I" * n_qubits:
        raw_results = np.array(circuit()).T
        for i, measure_z_vector in enumerate(raw_results):
            result_data.update({'{}'.format(i): measure_z_vector.tolist()})
    else:
        raise Exception('Nothing to run')
        # return 0

    # Get function return from measurements in Z according to sub_hamiltonian
    measurements = {}
    for sub_string in sub_hamiltonian:
        measurements[sub_string] = 0

    for measure_z_vector in raw_results:
        for sub_string in sub_hamiltonian:

            prod_function = 1
            for i, measure_z in enumerate(measure_z_vector):
                if main_string[i] != "I":
                    prod_function *= measure_z ** int(sub_string[i])

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
    Calculates the exact energy in a specific state using matrix algebra
    This function is basically used to test that the Pennylane circuit is correct

    :param qubit_hamiltonian: hamiltonian in qubits
    :param state_preparation_gates: list of gates in simulation library format that represents the state
    :return: the expectation value of the state given the hamiltonian
    """

    # Initialize circuit.
    n_qubits = count_qubits(qubit_hamiltonian)
    dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)])
    # add gates to circuit

    #print('hola')
    def circuit_function():
        for gate in state_preparation_gates:
            qml.apply(gate)
        return qml.state()

    state_preparation_gates_test = state_preparation_gates
    #print(state_preparation_gates_test)
    def circuit_function_test():
        for gate in state_preparation_gates_test                       :
            qml.apply(gate)
        return qml.expval(qml.PauliZ(0))
    #print(np.round(qml.matrix(circuit_function_test)().real, decimals=5))
    #exit()
    # create and run circuit
    circuit = qml.QNode(circuit_function, dev_unique_wires, analytic=None)
    state_vector = circuit()
    #circuit_test = qml.QNode(circuit_function_test, dev_unique_wires, analytic=None)
    #print(qml.draw(circuit_test, show_all_wires=True)())
    #exit()
    #No cambia el hamiltoniano solo pone I cuando no hay puerta y lo deja bonito y con el formato adecuado para poder
    #meterlo en los qubits
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

