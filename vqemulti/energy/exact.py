from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion.utils import count_qubits
import numpy as np
import scipy


def exact_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state.

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :return: exact energy
    """

    # Transform Hamiltonian to matrix representation
    sparse_hamiltonian = get_sparse_operator(hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for coefficient, operator in zip(coefficients, ansatz):

        # Get the operator matrix representation of the operator
        sparse_operator = coefficient * get_sparse_operator(operator, n_qubit)

        # Exponentiate the operator and update ket t
        ket = scipy.sparse.linalg.expm_multiply(sparse_operator, ket)

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparse_hamiltonian * ket).real

    return energy


def exact_adapt_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the gradient of the energy with respect to the coefficients for adapt VQE Wave function.
    To be used in the exact energy minimization function

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

    gradient = []

    def recurse(hbra, ket, term):

        if term > 0:
            operator = coefficients[-term] * get_sparse_operator(ansatz[-term], n_qubit)
            hbra = (scipy.sparse.linalg.expm_multiply(-operator, hbra.transpose().conj())).transpose().conj()
            ket = scipy.sparse.linalg.expm_multiply(-operator, ket)

        operator = get_sparse_operator(ansatz[-(term+1)], n_qubit)
        gradient.insert(0, 2 * hbra.dot(operator).dot(ket)[0, 0].real)

        if term < len(coefficients) - 1:
            recurse(hbra, ket, term + 1)

    recurse(hbra, ket, 0)

    return gradient


def exact_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the energy of the state prepared by applying an ansatz (of the
    type of the VQE protocol) to a reference state.

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :return: exact energy
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

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparse_hamiltonian * ket).real

    return energy


def exact_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the gradient of the energy with respect to the coefficients for VQE function.
    To be used in the exact energy minimization function

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
