from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion import FermionOperator
import numpy as np
import scipy


def get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orbitals, frozen_core=0):
    """
    Calculates the one particle density matrix in the molecular orbitals basis

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

def get_second_order_density_matrix(coefficients, ansatz, hf_reference_fock, n_orbitals, frozen_core=0):
    """
    Calculates the 2nd order density matrix in molecular orbitals basis

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
    density_matrix_alpha = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    density_matrix_beta = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    density_matrix_cross = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))

    for i in range(frozen_core):
        density_matrix_alpha[i, i, i, i] = 1
        density_matrix_beta[i, i, i, i] = 1

    # build density matrix
    for i1 in range(n_orbitals - frozen_core):
        for i2 in range(n_orbitals - frozen_core):

            for j1 in range(n_orbitals - frozen_core):
                for j2 in range(n_orbitals - frozen_core):

                    # alpha
                    density_operator = FermionOperator(((j1*2, 1), (j2*2, 1), (i2*2, 0), (i1*2, 0)))
                    #print(density_operator)
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_alpha[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] = np.sum(bra * sparse_operator * ket).real

                    # beta
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2+1, 1), (i2*2+1, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_beta[i1+frozen_core, i2+frozen_core,
                                        j2+frozen_core, j1+frozen_core] = np.sum(bra * sparse_operator * ket).real

                    # cross 1
                    density_operator = FermionOperator(((j1*2, 1), (j2*2+1, 1), (i2*2, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] -= np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                    # cross 2
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2, 1), (i2*2+1, 0), (i1*2, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] -= np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                   # cross 3
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2, 1), (i2*2, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] += np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                    # cross 4
                    density_operator = FermionOperator(((j1*2, 1), (j2*2+1, 1), (i2*2+1, 0), (i1*2, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] += np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

    return 0.5 * (3*density_matrix_alpha + 3*density_matrix_beta + density_matrix_cross)

def density_fidelity(density_ref, density_vqe):
    """
    compute quantum fidelity based in 1p-density matrices.
    source: https://en.wikipedia.org/wiki/Fidelity_of_quantum_states.
    The result is a real number between 0 (low fidelity) and 1 (high fidelity)

    :param density_ref: density matrix of reference wave function (usually fullCI)
    :param density_vqe: density matrix of target wave function (usually VQE)
    :return: quantum fidelity measure
    """

    n_electrons = np.trace(density_ref)

    density_ref = np.array(density_ref)/n_electrons
    density_vqe = np.array(density_vqe)/n_electrons

    # padding with zero if different sizes
    n_pad = len(density_ref) - len(density_vqe)
    if n_pad > 0:
        density_vqe = np.pad(density_vqe, (0, n_pad), 'constant')
    else:
        density_ref = np.pad(density_ref, (0, -n_pad), 'constant')

    sqrt_vqe = scipy.linalg.sqrtm(density_vqe)

    dens = np.dot(np.dot(sqrt_vqe, density_ref), sqrt_vqe)
    trace = np.trace(scipy.linalg.sqrtm(dens))

    # alternative
    # sqrt_ref = scipy.linalg.sqrtm(density_ref)
    # dens = np.dot(sqrt_ref, sqrt_vqe)
    # trace = np.trace(scipy.linalg.sqrtm(np.dot(dens.T, dens)))

    return np.absolute(trace)**2


def density_fidelity_simple(density_ref, density_vqe):
    """
    compute Quantum fidelity based in 1p-density matrices.
    Simple implementation defined as 1 - (the norm of the difference between density matrices)
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
