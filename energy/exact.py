import numpy as np
import scipy
from openfermion import get_sparse_operator
from utils import get_sparse_ket_from_fock
from openfermion.utils import count_qubits


def exact_vqe_energy(coefficients, ansatz, hf_reference_fock, qubit_hamiltonian):
    """
    Calculates the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state.
    Uses instances of scipy.sparse.csc_matrix for efficiency.

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param qubit_hamiltonian: Hamiltonian in qubits
    :return: exact energy
    """

    #ansatz = ansatz.copy()
    #ansatz.scale_vector(coefficients)

    #ansatz_qubit = ansatz.get_quibits_list()
    #coefficients = np.ones(len(ansatz_qubit))

    # Transform Hamiltonian to matrix representation
    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(qubit_hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for coefficient, operator in zip(coefficients, ansatz):

        # Get the operator matrix representation of the operator
        sparse_operator = get_sparse_operator(coefficient * operator, n_qubit)

        # Exponentiate the operator and update ket to represent the state after
        # this operator has been applied
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)

        ket = exp_operator * ket

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparse_hamiltonian * ket).real

    return energy

