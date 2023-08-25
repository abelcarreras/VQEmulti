from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion import FermionOperator
import numpy as np
import scipy


def get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orbitals, frozen_core=0):
    """
    Calculates the density matrix in molecular orbitals basis

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param n_orbitals: number of molecular orbitals
    :return: exact energy
    """

    # get number of qubits
    n_orbitals = n_orbitals
    n_qubit = (n_orbitals - frozen_core) * 2

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
    for i in range(frozen_core):
        density_matrix_alpha[i, i] = 1
        density_matrix_beta[i, i] = 1

    # build density matrix
    for i in range(n_orbitals - frozen_core):
        for j in range(n_orbitals - frozen_core):
            # alpha
            density_operator = FermionOperator(((j*2, 1), (i*2, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_alpha[i+frozen_core, j+frozen_core] = np.sum(bra * sparse_operator * ket).real

            # beta
            density_operator = FermionOperator(((j*2+1, 1), (i*2+1, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_beta[i+frozen_core, j+frozen_core] = np.sum(bra * sparse_operator * ket).real

    return density_matrix_alpha + density_matrix_beta


def density_fidelity(density_ref, density_vqe):
    """
    compute fidelity based in 1p-density matrices.
    The result is a real number between 0 (low fidelity) and 1 (high fidelity)

    :param density_ref: density matrix of reference wave function (usually fullCI)
    :param density_vqe: density matrix of target wave function (usually VQE)
    :return: fidelity measure
    """
    density_ref = np.array(density_ref)
    density_vqe = np.array(density_vqe)
    n_electrons = np.trace(density_ref)

    n_pad = len(density_ref) - len(density_vqe)
    fidelity = 1 - (np.linalg.norm(density_ref - np.pad(density_vqe, (0, n_pad), 'constant')) / n_electrons)

    return fidelity
