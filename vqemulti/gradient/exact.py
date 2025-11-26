import openfermion
from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
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
        sparse_operator = coefficient * get_sparse_operator(operator, n_qubits)
        state = scipy.sparse.linalg.expm_multiply(sparse_operator, state)
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


def compute_gradient_vector(hf_reference_fock, hamiltonian, ansatz, coefficients, pool):
    """
    computes the gradient vector of the energy with respect to the pool operators

    :param hf_reference_fock: reference HF state in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :param ansatz: VQE ansatz in qubit operators
    :param coefficients: list of VQE coefficients
    :param pool: pool of qubit operators
    :return: the gradient vector
    """

    # get n_qubits
    n_qubits = len(hf_reference_fock)

    # transform hamiltonian to sparse
    sparse_hamiltonian = get_sparse_operator(hamiltonian, n_qubits)

    # Prepare the current state from ansatz (& coefficient) and HF reference
    sparse_state = prepare_adapt_state(hf_reference_fock,
                                       ansatz,
                                       coefficients)

    # Calculate and print gradients
    # print('pool size: ', len(pool))
    print("\nNon-Zero Gradients (exact)")
    gradient_vector = []
    for i, operator in enumerate(pool):
        sparse_operator = get_sparse_operator(operator, n_qubits)
        gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)

        if gradient > 1e-5:
            print("Operator {}: {:.6f}".format(i, gradient))

        gradient_vector.append(gradient)

    return gradient_vector


def exact_adapt_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the gradient vector of the energy with respect to the coefficients for adapt VQE Wave function.
    To be used as gradient function in the exact energy minimization function in VQE

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :return: gradient vector
    """

    # Transform Hamiltonian to matrix representation
    sparse_hamiltonian = get_sparse_operator(hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for j, (coefficient, operator) in enumerate(zip(coefficients, ansatz)):

        # Get the operator matrix representation of the operator
        sparse_operator = coefficient * get_sparse_operator(operator, n_qubit)

        # Exponentiate the operator and update ket t
        ket = scipy.sparse.linalg.expm_multiply(sparse_operator, ket)

    bra = ket.transpose().conj()
    hbra = bra.dot(sparse_hamiltonian)

    gradient_vector = []

    def recurse(hbra, ket, term):

        if term > 0:
            operator = coefficients[-term] * get_sparse_operator(ansatz[-term], n_qubit)
            hbra = (scipy.sparse.linalg.expm_multiply(-operator, hbra.transpose().conj())).transpose().conj()
            ket = scipy.sparse.linalg.expm_multiply(-operator, ket)

        operator = get_sparse_operator(ansatz[-(term+1)], n_qubit)
        gradient_vector.insert(0, 2 * hbra.dot(operator).dot(ket)[0, 0].real)

        if term < len(coefficients) - 1:
            recurse(hbra, ket, term + 1)

    recurse(hbra, ket, 0)

    return gradient_vector


def exact_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the gradient vector of the energy with respect to the coefficients for VQE function.
    To be used as gradient function in the exact energy minimization function in adaptVQE

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :return: gradient vector
    """

    # Transform Hamiltonian to matrix representation
    sparse_hamiltonian = get_sparse_operator(hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Get total exponent operator list
    exponent = scipy.sparse.csr_array((2**n_qubit, 2**n_qubit), dtype=float)
    for coefficient, operator in zip(coefficients, ansatz):
        exponent += coefficient * get_sparse_operator(operator, n_qubit)

    # Apply operator to ket
    ket = scipy.sparse.linalg.expm_multiply(exponent, ket)

    bra = ket.transpose().conj()
    hbra = bra.dot(sparse_hamiltonian)

    gradient = []
    # Compute gradient as A_n * exp(A_1*c_1 + A_2*c_2 + A_3*c_3 + ... )
    for a_term, coefficient in zip(ansatz, coefficients):
        operator = get_sparse_operator(a_term, n_qubit)
        gradient.append(hbra.dot(operator).dot(ket)[0, 0].real)

    return gradient
