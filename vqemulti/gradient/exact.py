import openfermion
from openfermion import get_sparse_operator
from vqemulti.utils import get_sparse_ket_from_fock
from openfermion.utils import count_qubits
import numpy as np
import scipy


def prepare_adapt_state(hf_reference_fock, ansatz, coefficients):
    """
    prepare state from coefficients ansatz and reference

    :param hf_reference_fock:  reference HF state in Fock space vector
    :param ansatz: ansatz in qubit operators
    :param coefficients: ansatz operators scale coefficients
    :param n_qubit: number of q_bits
    :return:

    """

    # Initialize the state vector with the reference state.
    # |state> = |state_old> · exp(i * coef · |ansatz>
    # imaginary i already included in |ansatz>
    state = get_sparse_ket_from_fock(hf_reference_fock)

    n_qubits = len(hf_reference_fock)

    # Apply the ansatz operators one by one to obtain the state as optimized by the last iteration
    for coefficient, operator in zip(coefficients, ansatz):

        # Obtain the sparse matrix representing the operator
        sparse_operator = get_sparse_operator(coefficient * operator, n_qubits)

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

    :param sparse_operator:  the pool operator A in sparse matrix representation
    :param sparse_state: the state in which to calculate the energy (sparse vector representation)
    :param sparse_hamiltonian: the Hamiltonian of the system in sparse matrix representation
    :return: gradient (float)
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
    """
    computes the gradient vector respect to the pool operators

    :param hf_reference_fock: reference HF state in Fock space vector
    :param qubit_hamiltonian: hamiltonian in qubit operators
    :param ansatz: VQE ansatz in qubit operators
    :param coefficients: list of VQE coefficients
    :param pool: pool of qubit operators
    :return: the gradient vector
    """

    # transform hamiltonian to sparse
    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
    n_qubits = count_qubits(qubit_hamiltonian)

    # Prepare the current state from ansatz (& coefficient) and HF reference
    sparse_state = prepare_adapt_state(hf_reference_fock,
                                       ansatz,
                                       coefficients)

    # Calculate and print gradients
    print('pool size: ', len(pool))
    print("Non-Zero Gradients (calculated)")
    gradient_vector = []
    for i, operator in enumerate(pool):
        sparse_operator = get_sparse_operator(operator, n_qubits)
        gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)

        if gradient > 1e-5:
            print("Operator {}: {:.6f}".format(i, gradient))

        gradient_vector.append(gradient)

    return gradient_vector


