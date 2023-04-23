from openfermion import get_sparse_operator
from utils import get_sparse_ket_from_fock
from openfermion.utils import count_qubits
import numpy as np
import scipy


def prepare_adapt_state(reference_state, ansatz, coefficients, n_qubit):
    # Initialize the state vector with the reference state.
    # |state> = |state_old> · exp(i * coef · |ansatz>
    # imaginary i already included in |ansatz> by generate_jw_operator_pool function

    state = reference_state.copy()

    # Apply the ansatz operators one by one to obtain the state as optimized
    # by the last iteration
    for coefficient, operator in zip(coefficients, ansatz):
        # Multiply the operator by the variational parameter
        operator = coefficient * operator

        # Obtain the sparse matrix representing the operator
        sparse_operator = get_sparse_operator(operator, n_qubit)

        # Exponentiate the operator
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)

        # Act on the state with the operator
        state = exp_operator.dot(state)

    return state


def calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian):
    """
    Given an operator A, calculates the gradient of the energy with respect to the
    coefficient c of the operator exp(c * A), at c = 0, in a given state.
    Uses dexp(c*A)/dc = <psi|[H,A]|psi> = 2 * real(<psi|HA|psi>)

    Arguments:
      sparse_operator (scipy.sparse.csc_matrix): the pool operator A.
      sparse_state (scipy.sparse.csc_matrix): the state in which to calculate the energy.
      sparse_hamiltonian (scipy.sparse.csc_matrix): the Hamiltonian of the system.

    Returns:
      gradient (float): the norm of dE/dc in this state, at point c = 0.
    """

    # gradient 2 * <state | H · Op | state >  (non-explicit)
    bra = sparse_state.transpose().conj()
    ket = sparse_operator.dot(sparse_state)
    gradient = 2 * np.abs(bra * sparse_hamiltonian * ket)[0, 0].real

    # < state | [H , Op] | state > (explicit)
    # commutator = sparseHamiltonian.dot(sparseOperator) - sparseOperator.dot(sparseHamiltonian)
    # gradient = np.sum(state.transpose().conj().dot(commutator).dot(state)).real

    return gradient


def compute_gradient_vector(hf_reference_fock, qubit_hamiltonian, ansatz, coefficients, pool):

    # transform Fock state to sparse (and ket by transposing)
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

    # Calculate and print gradients
    print("Non-Zero Gradients (calculated)")
    gradient_vector = []
    for i, operator in enumerate(pool):
        sparse_operator = get_sparse_operator(operator, n_qubits)
        gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)

        if gradient > 1e-5:
            print("Operator {}: {}".format(i, gradient))

        gradient_vector.append(gradient)

    return gradient_vector


