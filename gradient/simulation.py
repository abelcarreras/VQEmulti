from utils import get_sparse_ket_from_fock, convert_hamiltonian, group_hamiltonian
from gradient.exact import prepare_adapt_state
from energy.simulation.tools import measureExpectation, get_exact_state_evaluation, build_gradient_ansatz
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import numpy as np
import scipy


def get_sampled_gradient(qubitOperator, qubitHamiltonian, statePreparationGates, n_qubits, shots=1000):
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
      n_qubits (int): number of qubits.
      shots (int): number of shoots

    Returns:
      commutator (float): the sampled expectation value of <psi|[H,A]|psi>, that
        is an estimate of the value of the gradient.
    """

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
                                               n_qubits)
        commutator += expectation_value

    assert commutator.imag < 1e-5

    return commutator.real


def simulate_gradient(hf_reference_fock, qubit_hamiltonian, ansatz, coefficients, pool, shots=10000, sample=True):

    # transform reference to sparse (and ket by transposing) using Jordan-Wigner transform
    sparse_reference_state = get_sparse_ket_from_fock(hf_reference_fock)

    # qubit hamiltonian to sparse (Jordan-Wigner transform is used)
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

    total_op_matrix = identity
    for _ in range(n_qubits - 1):
        total_op_matrix = scipy.sparse.kron(identity, total_op_matrix, 'csc')

    # Multiply the identity matrix by the matrix form of each operator in the
    # ansatz, to obtain the matrix representing the action of the complete ansatz
    for coefficient, operator in zip(coefficients, ansatz):
        # Get corresponding the sparse operator, with the correct dimension
        # (forcing n_qubits = qubitNumber, even if this operator acts on less qubits)
        operatorMatrix = get_sparse_operator(coefficient * operator, n_qubits)

        # Multiply previous matrix by this operator
        total_op_matrix = scipy.sparse.linalg.expm(operatorMatrix) * total_op_matrix

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(pool):
        print("Operator {}".format(i))

        # Prepare state from HF reference and total operator matrix
        state_preparation_gates = build_gradient_ansatz(hf_reference_fock, total_op_matrix)

        if sample:
            sampled_gradient = np.abs(get_sampled_gradient(operator,
                                                           qubit_hamiltonian,
                                                           state_preparation_gates,
                                                           n_qubits,
                                                           shots=shots))
        else:
            # Calculate the exact energy in this state
            commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian
            sampled_gradient = np.abs(get_exact_state_evaluation(commutator_hamiltonian, state_preparation_gates).real)
            print('Exact gradient (simulator): {:.6f}'.format(sampled_gradient))

        # compute exact gradient for comparison (can be omitted if needed)
        from gradient.exact import calculate_gradient
        sparse_operator = get_sparse_operator(operator, n_qubits)
        calculated_gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)
        print("Exact gradient: {:.6f}".format(calculated_gradient))

        error = np.abs(sampled_gradient - calculated_gradient)
        print("simulated gradient: {:.6f} ({} shots)".format(sampled_gradient, shots))

        if abs(calculated_gradient) > 1e-3:
            print("Error: {0:.3f} ({1:.3f}%)\n".format(error, 100*error/calculated_gradient))
        else:
            print("Error: {0:.3f} NA%)\n".format(error))

        gradient_vector.append(sampled_gradient)

    return gradient_vector
