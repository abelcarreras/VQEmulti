from energy.simulation.tools_penny import build_reference_gates
import numpy as np
import pennylane as qml


def trotter_step(operator, time):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.

    :param operator: qubit operator
    :param time: the evolution time
    :return: trotter_gates
    """

    # Initialize list of gates
    trotter_gates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    ordered_terms = sorted(list(operator.terms.keys()))

    # Add to trotter_gates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauliString in ordered_terms:

        # Get real part of the coefficient (the immaginary one can't be simulated,
        # as the exponent would be real and the operation would not be unitary).
        # Multiply by time to get the full multiplier of the Pauli string.
        coefficient = float(np.real(operator.terms[pauliString])) * time

        # Keep track of the qubit indices involved in this particular Pauli string.
        # It's necessary so as to know which are included in the sequence of CNOTs
        # that compute the parity
        involved_qubits = []

        # Perform necessary basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]
            involved_qubits.append(qubit_index)

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to X basis
                trotter_gates.append(qml.Hadamard(wires= qubit_index))

            if pauli_operator == "Y":
                # Rotate to Y Basis
                trotter_gates.append(qml.RX(np.pi / 2,wires= qubit_index))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(qml.CNOT(wires= [control, target]))

        # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits)
        trotter_gates.append(qml.RZ((2 * coefficient), wires= last_qubit))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(qml.CNOT(wires= [control, target]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to Z basis from X basis
                trotter_gates.append(qml.Hadamard(qubit_index))

            if pauli_operator == "Y":
                # Rotate to Z basis from Y Basis
                trotter_gates.append(qml.RX(-np.pi / 2, wires = qubit_index))

    return trotter_gates


def trotterize_operator(operator, time, trotter_steps):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator', with the given
    number of steps.
    Increasing the number of steps increases precision (unless the terms in the
    operator commute, in which case steps = 1 is already exact).
    For the same precision, a greater time requires a greater step number
    (again, unless the terms commute)

    :param operator: qubit operator
    :param time: the evolution time
    :param trotter_steps: number of trotter steps
    :return: the number of trotter steps to split the time evolution into
    """

    # Divide time into steps and apply the evolution operator the necessary
    # number of times
    trotter_gates = []
    for step in range(1, trotter_steps + 1):
        trotter_gates += trotter_step(operator, time / trotter_steps)

    return trotter_gates


def get_preparation_gates_trotter(coefficients, ansatz_qubit, trotter_steps, hf_reference_fock):
    """
    Trotterize the ansatz

    :param coefficients: ansatz coefficients
    :param ansatz_qubit: operators list in qubit
    :param trotter_steps: number of trotter steps
    :param hf_reference_fock: reference HF in Fock vspace vector
    :return: trotterized gates list
    """

    # Initialize the ansatz gate list
    trotter_ansatz = []

    # Go through the operators in the ansatz
    for coefficient, operator in zip(coefficients, ansatz_qubit):
        # Get the trotterized circuit for applying e**(operator*coefficient)
        operator_trotter_circuit = trotterize_operator(1j*operator,
                                                       coefficient,
                                                       trotter_steps)

        # Add the gates corresponding to this operator to the ansatz gate list
        trotter_ansatz += operator_trotter_circuit

    # Initialize the state preparation gates with the reference state preparation gates
    state_preparation_gates = build_reference_gates(hf_reference_fock)

    # return total trotterized ansatz
    return state_preparation_gates + trotter_ansatz
