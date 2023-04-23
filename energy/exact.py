import numpy as np
import scipy
from openfermion import get_sparse_operator
from utils import get_sparse_ket_from_fock
from openfermion.utils import count_qubits


def exact_vqe_energy(coefficients, operators, hf_reference_fock, qubitHamiltonian):
    '''
    Calculates the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state.
    Uses instances of scipy.sparse.csc_matrix for efficiency.

    Arguments:
      coefficients ([float]): the list of coefficients of the ansatz operators.
      operators (union[openfermion.FermionOperator,openfermion.QubitOperator]):
        the list of ansatz operators (fermionic ladder operators, pre
        exponentiation). The index of the operators in the list should be in
        accordance with the index of the respective coefficients in coefVect.
      referenceState (numpy.ndarray): the state vector representing the state
        prior to the application of the ansatz (e.g. the Hartree Fock ground
        state).
      sparseHamiltonian (scipy.sparse.csc_matrix): the hamiltonian of the system,
        as a sparse matrix.

      Returns:
        energy (float): the expectation value of the Hamiltonian in the state
          prepared by applying the ansatz to the reference state.
    '''

    sparseHamiltonian = get_sparse_operator(qubitHamiltonian)

    # Find the number of qubits of the system (2**qubitNumber = dimension)
    qubitNumber = count_qubits(qubitHamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for coefficient, operator in zip(coefficients, operators):
        # Multiply the operator by the respective coefficient
        operator = coefficient * operator

        # Get the sparse matrix representing the operator. The dimension should be
        # that of the FULL Hilbert space, even if the operator only acts on part.
        sparseOperator = get_sparse_operator(operator, qubitNumber)

        # Exponentiate the operator and update ket to represent the state after
        # this operator has been applied
        expOperator = scipy.sparse.linalg.expm(sparseOperator)

        # print('ket shape', ket.shape)
        # print('operator shape', expOperator.shape)

        ket = expOperator * ket

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparseHamiltonian * ket).real

    return energy

