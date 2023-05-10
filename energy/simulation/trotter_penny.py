from energy.simulation.tools_penny import build_reference_gates
import numpy as np
import pennylane as qml


def trotterStep(operator, qubitNumber, time):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.

    :param operator:
    :param qubitNumber:
    :param time:
    :return:
    """
    #print('trotter-step', operator)
    # Initialize list of gates
    trotterGates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    orderedTerms = sorted(list(operator.terms.keys()))

    # Add to trotterGates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauliString in orderedTerms:

        # Get real part of the coefficient (the immaginary one can't be simulated,
        # as the exponent would be real and the operation would not be unitary).
        # Multiply by time to get the full multiplier of the Pauli string.
        coef = float(np.real(operator.terms[pauliString])) * time

        # Keep track of the qubit indices involved in this particular Pauli string.
        # It's necessary so as to know which are included in the sequence of CNOTs
        # that compute the parity
        involvedQubits = []

        # Perform necessary basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubitIndex = pauli[0]
            involvedQubits.append(qubitIndex)

            # Get the Pauli operator identifier (X,Y or Z)
            pauliOp = pauli[1]

            if pauliOp == "X":
                # Rotate to X basis
                trotterGates.append(qml.Hadamard(wires= qubitIndex))

            if pauliOp == "Y":
                # Rotate to Y Basis
                trotterGates.append(qml.RX(np.pi / 2,wires= qubitIndex))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involvedQubits) - 1):
            control = involvedQubits[i]
            target = involvedQubits[i + 1]

            trotterGates.append(qml.CNOT(wires= [control, target]))

        # Apply e^(-i*Z*coef) = Rz(coef*2) to the last involved qubit
        lastQubit = max(involvedQubits)
        trotterGates.append(qml.RZ((2 * coef), wires= lastQubit))

        # Uncompute parity
        for i in range(len(involvedQubits) - 2, -1, -1):
            control = involvedQubits[i]
            target = involvedQubits[i + 1]

            trotterGates.append(qml.CNOT(wires= [control, target]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubitIndex = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauliOp = pauli[1]

            if pauliOp == "X":
                # Rotate to Z basis from X basis
                trotterGates.append(qml.Hadamard(qubitIndex))

            if pauliOp == "Y":
                # Rotate to Z basis from Y Basis
                trotterGates.append(qml.RX(-np.pi / 2, wires = qubitIndex))

    return trotterGates


def trotterizeOperator(operator, qubitNumber, time, steps):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator', with the given
    number of steps.
    Increasing the number of steps increases precision (unless the terms in the
    operator commute, in which case steps = 1 is already exact).
    For the same precision, a greater time requires a greater step number
    (again, unless the terms commute)

    Arguments:
      operator (union[openfermion.QubitOperator, openfermion.FermionOperator,
        openfermion.InteractionOperator]): the operator to be simulated
      qubits ([cirq.LineQubit]): the qubits that the gates should be applied to
      time (float): the evolution time
      steps (int): the number of trotter steps to split the time evolution into

    Returns:
      trotterGates : the list of gates that apply the
        trotterized operator
    """
    #print('trotopt', operator)
    # Divide time into steps and apply the evolution operator the necessary
    # number of times
    trotterGates = []
    for step in range(1, steps + 1):
        trotterGates += trotterStep(operator, qubitNumber, time / steps)

    return trotterGates


def get_preparation_gates_trotter(coefficients, ansatz, trotter_steps, hf_reference_fock):
    """
    trotterize the operators

    :param coefficients:
    :param ansatz:
    :param trotter_steps:
    :param hf_reference_fock:
    :param n_qubits:
    :return: gates list
    """
    #print('get-prep-gates', ansatz)
    n_qubits = len(hf_reference_fock)
    # Initialize the ansatz gate list
    trotter_ansatz = []
    # Go through the operators in the ansatz
    for coefficient, fermionOperator in zip(coefficients, ansatz):
        # Get the trotterized circuit for applying e**(operator*coefficient)
        operator_trotter_circuit = trotterizeOperator(1j * fermionOperator,
                                                      n_qubits,
                                                      coefficient,
                                                      trotter_steps)

        # Add the gates corresponding to this operator to the ansatz gate list
        trotter_ansatz += operator_trotter_circuit
    #print('result', trotter_ansatz)
    #exit()
    # Initialize the state preparation gates with the reference state preparation gates
    state_preparation_gates = build_reference_gates(hf_reference_fock)

    # Append the trotterized ansatz
    state_preparation_gates_final = state_preparation_gates + trotter_ansatz

    return state_preparation_gates_final
