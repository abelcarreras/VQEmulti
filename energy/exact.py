import numpy as np
import scipy
from openfermion import get_sparse_operator
from utils import get_sparse_ket_from_fock
from openfermion.utils import count_qubits


def exact_vqe_energy(coefficients, operators, hf_reference_fock, qubit_hamiltonian):
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
      hf_reference_fock ([int]): the state vector representing the state
        prior to the application of the ansatz (e.g. the Hartree Fock ground
        state).
      qubit_hamiltonian (openfermion qubit): the hamiltonian of the system,
        as a qbits.

      Returns:
        energy (float): the expectation value of the Hamiltonian in the state
          prepared by applying the ansatz to the reference state.
    '''

    # Transform Hamiltonian to matrix representation (JW transformation)
    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(qubit_hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix (JW transformation)
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for coefficient, operator in zip(coefficients, operators):
        # Multiply the operator by the respective coefficient
        operator = coefficient * operator

        # Get the operator matrix representation of the operator (JW)
        sparse_operator = get_sparse_operator(operator, n_qubit)

        # Exponentiate the operator and update ket to represent the state after
        # this operator has been applied
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)

        # print('ket shape', ket.shape)
        # print('operator shape', exp_operator.shape)
        ket = exp_operator * ket

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparse_hamiltonian * ket).real

    return energy

