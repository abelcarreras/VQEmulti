from utils import convert_hamiltonian, group_hamiltonian
from energy.simulation.tools import trotterizeOperator, measureExpectation, get_exact_state_evaluation
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import cirq
import scipy


def get_sampled_energy(qubitHamiltonian, shots, statePreparationGates):
    '''
    Obtains the expectation value in a state by sampling (using the CIRQ
    simulator).

    Arguments:
      groupedHamiltonian (dict): the Hamiltonian, grouped by terms that only differ
        by identities (as done by groupHamiltonian).
      shots (int): the number of circuit repetitions to be used.
      statePreparationGates (list): a list of CIRQ gates that prepare the state.
      qubits (list): a list of cirq.LineQubit to apply the gates on.

    Returns:
      energy (float): the energy (with sampling noise).
    '''

    qubitNumber = count_qubits(qubitHamiltonian)
    qubits = cirq.LineQubit.range(qubitNumber)

    print(qubitHamiltonian)
    formattedHamiltonian = convert_hamiltonian(qubitHamiltonian)
    print(formattedHamiltonian)
    groupedHamiltonian = group_hamiltonian(formattedHamiltonian)
    for group in groupedHamiltonian:
        print(group, groupedHamiltonian[group])

    # Obtain the experimental expectation value for each Pauli string by
    # calling the measureExpectation function, and perform the necessary weighed
    # sum to obtain the energy expectation value

    energy = 0
    for main_string, sub_hamiltonian in groupedHamiltonian.items():
        expectation_value = measureExpectation(main_string,
                                               sub_hamiltonian,
                                               shots,
                                               statePreparationGates,
                                               qubits)
        energy += expectation_value

    assert energy.imag < 1e-5

    return energy.real


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, qubitHamiltonian, shots,
                        trotter=True, trotter_steps=1, sample=True):
    """
    Obtains the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state, using the CIRQ simulator.

    Arguments:
      coefficients ([float]): the list of coefficients of the ansatz operators.
      ansatz ([openfermion.FermionOperator]): the list of ansatz operators
        (fermionic ladder operators, pre exponentiation). The index of the
        operators in the list should be in accordance with the index of the
        respective coefficients in coefVect.
      shots (int): the number of circuit repetitions to use to get the
        expectation value. The precision of the estimate goes like 1/sqrt(shots)
      trotter (bool): whether to trotterize the ansatz. If False, the corresponding
        matrix will be applied directly using a cirq.MatrixGate.
      trotter_steps (int): the number of trotter steps to be used in the trotterization.
      sample (bool): whether to sample from the circuit. If False, the full
        wavevector will be simulated, and the final energy will be calculated
        analytically.

      Returns:
        energy (float): the expectation value of the Hamiltonian in the state
          prepared by applying the ansatz to the reference state, with trotter
          error (if trotter = True) and sampling noise (if sample = True)
    """

    # Format and group the Hamiltonian, so as to save measurements by using
    # the same data for Pauli strings that only differ by identities
    # formattedHamiltonian = convertHamiltonian(qubitHamiltonian)
    # groupedHamiltonian = groupHamiltonian(formattedHamiltonian)

    # Count the number of qubits the Hamiltonian acts on
    qubitNumber = count_qubits(qubitHamiltonian)

    def build_reference_gates(hf_reference_fock, qubitNumber):
        # Initialize qubits
        qubits = cirq.LineQubit.range(qubitNumber)

        # Create the gates for preparing the Hartree Fock ground state, that serves
        # as a reference state the ansatz will act on
        return [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    def get_preparation_gates_trotter(coefficients, ansatz, trotter_steps):

        # If the trotter flag is set, trotterize the operators into a CIRQ circuit
        #if trotter:

        # Initialize the ansatz gate list
        trotterAnsatz = []

        qubits = cirq.LineQubit.range(qubitNumber)

        # Go through the operators in the ansatz
        for coefficient, fermionOperator in zip(coefficients, ansatz):
            # Get the trotterized circuit for applying e**(operator*coefficient)
            operatorTrotterCircuit = trotterizeOperator(1j * fermionOperator,
                                                        qubits,
                                                        coefficient,
                                                        trotter_steps)

            # Add the gates corresponding to this operator to the ansatz gate list
            trotterAnsatz += operatorTrotterCircuit

        # Initialize the state preparation gates with the reference state preparation
        # gates
        hf_reference_gates = build_reference_gates(hf_reference_fock, qubitNumber)
        statePreparationGates = hf_reference_gates

        # Append the trotterized ansatz
        statePreparationGates.append(trotterAnsatz)
        return statePreparationGates

    def get_preparation_gates(coefficients, ansatz):

        qubits = cirq.LineQubit.range(qubitNumber)

        # Create sparse 2x2 identity matrix and initialize the ansatz matrix with it
        identity = scipy.sparse.identity(2, format='csc', dtype=complex)

        # Multiply the ansatz matrix by identity as many times as necessary to get
        # the correct dimension
        matrix = identity
        for _ in range(qubitNumber - 1):
            matrix = scipy.sparse.kron(identity, matrix, 'csc')

        # Multiply the identity matrix by the matrix form of each operator in the
        # ansatz, to obtain the matrix representing the action of the complete ansatz
        for (coefficient, operator) in zip(coefficients, ansatz):
            # Get corresponding the sparse operator, with the correct dimension
            # (forcing n_qubits = qubitNumber, even if this operator acts on less
            # qubits)
            operatorMatrix = get_sparse_operator(coefficient * operator, qubitNumber)

            # Multiply previous matrix by this operator
            matrix = scipy.sparse.linalg.expm(operatorMatrix) * matrix

        # Initialize the state preparation gates with the Hartree Fock preparation
        # circuit
        statePreparationGates = build_reference_gates(hf_reference_fock, qubitNumber)

        # Append the ansatz directly as a matrix
        statePreparationGates.append(cirq.MatrixGate(matrix.toarray()).on(*qubits))

        return statePreparationGates


    # If the trotter flag isn't set, use matrix exponentiation to get the exact
    # matrix representing the action of the ansatz, and apply it directly on the
    if trotter:
        statePreparationGates = get_preparation_gates_trotter(coefficients, ansatz, trotter_steps)
    else:
        statePreparationGates = get_preparation_gates(coefficients, ansatz)

    if sample:
        # Obtain the energy expectation value by sampling from the circuit using
        # the CIRQ simulator
        energy = get_sampled_energy(qubitHamiltonian, shots, statePreparationGates)
    else:
        # Calculate the exact energy in this state
        energy = get_exact_state_evaluation(qubitHamiltonian, statePreparationGates)
        # print('Exact energy ener (no sample): ', energy)

    return energy
