from utils import convert_hamiltonian, string_to_matrix
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.utils import count_qubits
from openfermion import InteractionOperator, FermionOperator
import numpy as np
import cirq


def trotterStep(operator, qubits, time):
    '''
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.

    Arguments:
      operator (union[openfermion.QubitOperator, openfermion.FermionOperator,
        openfermion.InteractionOperator]): the operator to be simulated
      qubits ([cirq.LineQubit]): the qubits that the gates should be applied to
      time (float): the evolution time

    Returns:
      trotterGates (cirq.OP_TREE): the list of CIRQ gates that apply the
        trotterized operator
    '''

    # If operator is an InteractionOperator, shape it into a FermionOperator
    if isinstance(operator, InteractionOperator):
        operator = get_fermion_operator(operator)

    # If operator is a FermionOperator, use the Jordan Wigner transformation
    # to map it into a QubitOperator
    if isinstance(operator, FermionOperator):
        operator = jordan_wigner(operator)

    # Get the number of qubits the operator acts on
    qubitNumber = count_qubits(operator)

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
                trotterGates.append(cirq.H(qubits[qubitIndex]))

            if pauliOp == "Y":
                # Rotate to Y Basis
                trotterGates.append(cirq.rx(np.pi / 2).on(qubits[qubitIndex]))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involvedQubits) - 1):
            control = involvedQubits[i]
            target = involvedQubits[i + 1]

            trotterGates.append(cirq.CX(qubits[control], qubits[target]))

        # Apply e^(-i*Z*coef) = Rz(coef*2) to the last involved qubit
        lastQubit = max(involvedQubits)
        trotterGates.append(cirq.rz(2 * coef).on(qubits[lastQubit]))

        # Uncompute parity
        for i in range(len(involvedQubits) - 2, -1, -1):
            control = involvedQubits[i]
            target = involvedQubits[i + 1]

            trotterGates.append(cirq.CX(qubits[control], qubits[target]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubitIndex = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauliOp = pauli[1]

            if pauliOp == "X":
                # Rotate to Z basis from X basis
                trotterGates.append(cirq.H(qubits[qubitIndex]))

            if pauliOp == "Y":
                # Rotate to Z basis from Y Basis
                trotterGates.append(cirq.rx(-np.pi / 2).on(qubits[qubitIndex]))

    return trotterGates


def trotterizeOperator(operator, qubits, time, steps):
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
      trotterGates (cirq.OP_TREE): the list of CIRQ gates that apply the
        trotterized operator
    """


    # Divide time into steps and apply the evolution operator the necessary
    # number of times
    trotterGates = []
    for step in range(1, steps + 1):
        trotterGates += (trotterStep(operator, qubits, time / steps))

    return trotterGates


def measureExpectation(main_string, sub_hamiltonian, shots, statePreparationGates, qubits):
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
      statePreparationGates (list): the list of CIRQ gates that prepare (from
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
    circuit.append(statePreparationGates)

    # cirq.optimizers.EjectZ().optimize_circuit(circuit)
    # cirq.optimizers.DropNegligible().optimize_circuit(circuit)

    # optimize circuit
    circuit = cirq.eject_z(circuit)
    circuit = cirq.drop_negligible_operations(circuit)

    n_qubits = len(qubits)

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
    total = {}
    for sub_string in sub_hamiltonian:
        total[sub_string] = 0

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
            total[sub_string] += meas[sub_string]

    # Calculate the expectation value of the subHamiltonian, by multiplying
    # the expectation value of each substring by the respective coefficient
    total_expectation_value = 0
    for sub_string in sub_hamiltonian:
        # Get the expectation value of this substring by taking the average
        # over all the repetitions
        expectation_value = total[sub_string] / shots

        # Add this value to the total expectation value, weighed by its
        # coefficient
        total_expectation_value += expectation_value * sub_hamiltonian[sub_string]

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


