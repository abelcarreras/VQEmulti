from utils import get_sparse_ket_from_fock, convert_hamiltonian, group_hamiltonian
from gradient.exact import prepare_adapt_state
from energy.simulation.tools import measureExpectation, get_exact_state_evaluation
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import numpy as np
import scipy
import cirq


# copy data from sampleEnergy TODO: check this
def get_sampled_gradient(qubitOperator, qubitHamiltonian, statePreparationGates, qubits, shots=1000):
    """
    Given an operator A, samples (using the CIRQ simulator) the gradient of the
    energy with respect to the coefficient c of the operator exp(c * A), at c = 0,
    in a given state.
    Uses dexp(c*A)/dc = <psi|[H,A]|psi>.

    Arguments:
      operator (Union[openfermion.QubitOperator, openfermion.FermionOperator,
        openfermion.InteractionOperator]): the pool operator A.
      hamiltonian (Union[openfermion.QubitOperator, openfermion.FermionOperator,
        openfermion.InteractionOperator]): the hamiltonian of the system.
      statePreparationGates  (list): a list of CIRQ gates that prepare the state
        in which to calculate the gradient.
      qubits (list): a list of cirq.LineQubit to apply the gates on.

    Returns:
      commutator (float): the sampled expectation value of <psi|[H,A]|psi>, that
        is an estimate of the value of the gradient.
    """

    qubitNumber = count_qubits(qubitHamiltonian)

    # < state| [H, A] |state>
    commutator_hamiltonian = qubitHamiltonian * qubitOperator - qubitOperator * qubitHamiltonian
    formattedCommutator = convert_hamiltonian(commutator_hamiltonian)
    groupedCommutator = group_hamiltonian(formattedCommutator)

    # Obtain the experimental expectation value for each Pauli string by
    # calling the measureExpectation function, and perform the necessary weighed
    # sum to obtain the energy expectation value
    commutator = 0
    for main_string, sub_hamiltonian in groupedCommutator.items():
        expectation_value = measureExpectation(main_string,
                                               sub_hamiltonian,
                                               shots,
                                               statePreparationGates,
                                               qubitNumber)
        commutator += expectation_value

    assert commutator.imag < 1e-5

    return commutator.real


def simulate_gradient(hf_reference_fock, qubit_hamiltonian, ansatz, coefficients, pool, shots=10000, sample=True):
    # Choose operator to use in the tests

    # transform reference to sparse (and ket by transposing)
    sparse_reference_state = get_sparse_ket_from_fock(hf_reference_fock)

    # transform hamiltonian to sparse
    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
    n_qubits = count_qubits(qubit_hamiltonian)


    # Get the current state.
    # Since the coefficients are reoptimized every iteration, the state has to
    # be built from the reference state each time.
    sparse_state = prepare_adapt_state(sparse_reference_state,
                                       ansatz,
                                       coefficients,
                                       n_qubits)

    identity = scipy.sparse.identity(2, format='csc', dtype=complex)

    matrix = identity
    for _ in range(n_qubits - 1):
        matrix = scipy.sparse.kron(identity, matrix, 'csc')

    # Multiply the identity matrix by the matrix form of each operator in the
    # ansatz, to obtain the matrix representing the action of the complete ansatz
    for (coefficient, operator) in zip(coefficients, ansatz):
        # Get corresponding the sparse operator, with the correct dimension
        # (forcing n_qubits = qubitNumber, even if this operator acts on less
        # qubits)
        operatorMatrix = get_sparse_operator(coefficient * operator, n_qubits)

        # Multiply previous matrix by this operator
        matrix = scipy.sparse.linalg.expm(operatorMatrix) * matrix

    qubits = cirq.LineQubit.range(n_qubits)
    gate_hf_reference = [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(pool):

        print("Operator {}".format(i))

        # compute exact gradient for comparison
        from gradient.exact import calculate_gradient
        sparse_operator = get_sparse_operator(operator, n_qubits)
        calculated_gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)
        print("Exact gradient: {:.6f}".format(calculated_gradient))

        # Add ansatz to HF reference to prepare state
        gate_hf_state = gate_hf_reference.copy()
        gate_hf_state.append(cirq.MatrixGate(matrix.toarray()).on(*qubits))

        if sample:
            sampled_gradient = np.abs(get_sampled_gradient(operator,
                                                           qubit_hamiltonian,
                                                           gate_hf_state,
                                                           qubits,
                                                           shots=shots))
        else:
            # Calculate the exact energy in this state
            commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian
            sampled_gradient = np.abs(get_exact_state_evaluation(commutator_hamiltonian, gate_hf_state).real)
            print('Exact gradient (simulator): {:.6f}'.format(sampled_gradient))

        error = np.abs(sampled_gradient - calculated_gradient)
        print("simulated gradient: {:.6f} ({} shots)".format(sampled_gradient, shots))

        if abs(calculated_gradient) > 1e-3:
            print("Error: {0:.3f} ({1:.3f}%)".format(error, 100*error/calculated_gradient))
        else:
            print("Error: {0:.3f} NA%)".format(error))

        print()
        gradient_vector.append(sampled_gradient)

    return gradient_vector
