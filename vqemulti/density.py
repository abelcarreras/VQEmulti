from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion import FermionOperator
import numpy as np
import scipy


def get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orbitals):
    """
    Calculates the density matrix in molecular orbitals basis

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param n_orbitals: number of molecular orbitals
    :return: exact energy
    """

    # get number of qubits
    n_qubit = n_orbitals * 2

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Get total exponent operator list
    exponent = scipy.sparse.csr_array((2 ** n_qubit, 2 ** n_qubit), dtype=float)
    for coefficient, operator in zip(coefficients, ansatz):
        exponent += coefficient * get_sparse_operator(operator, n_qubit)

    # Apply operator to ket
    ket = scipy.sparse.linalg.expm_multiply(exponent, ket)

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()

    # initialize density matrices
    density_matrix_alpha = np.zeros((n_orbitals, n_orbitals))
    density_matrix_beta = np.zeros((n_orbitals, n_orbitals))

    # build density matrix
    for i in range(n_orbitals):
        for j in range(n_orbitals):

            # alpha
            density_operator = FermionOperator(((j*2, 1), (i*2, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_alpha[i, j] = np.sum(bra * sparse_operator * ket).real

            # beta
            density_operator = FermionOperator(((j*2+1, 1), (i*2+1, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_beta[i, j] = np.sum(bra * sparse_operator * ket).real

    return density_matrix_alpha + density_matrix_beta
