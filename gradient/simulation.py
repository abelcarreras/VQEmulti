from utils import transform_to_scaled_qubit, fermion_to_qubit
from openfermion import get_sparse_operator, QubitOperator, count_qubits
import numpy as np


def print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient):
    """
    compute exact gradient using matrix representation for comparison
    qubit to sparse using Jordan-Wigner transform

    :param qubit_hamiltonian:
    :param hf_reference_fock:
    :param ansatz_qubit:
    :param operator:
    :param sampled_gradient:
    :return:
    """
    from gradient.exact import calculate_gradient, prepare_adapt_state

    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)  # Get the current hamiltonian.
    sparse_state = prepare_adapt_state(hf_reference_fock, ansatz_qubit,
                                       [1] * len(ansatz_qubit))  # Get the current state.
    sparse_operator = get_sparse_operator(operator, count_qubits(qubit_hamiltonian))
    calculated_gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)
    print("Exact gradient: {:.6f}".format(calculated_gradient))

    error = np.abs(sampled_gradient - calculated_gradient)

    if abs(calculated_gradient) > 1e-3:
        print("Error: {0:.3f} ({1:.3f}%)\n".format(error, 100 * error / calculated_gradient))
    else:
        print("Error: {0:.3f} NA%)\n".format(error))


def simulate_gradient(hf_reference_fock, qubit_hamiltonian, ansatz, coefficients, pool, simulator):
    """
    simulate gradient using quantum computer simulators (Cirq, pennylane)

    :param hf_reference_fock:  reference HF state in Fock space vector
    :param qubit_hamiltonian: hamiltonian in qubit operators
    :param ansatz: VQE ansatz in qubit/Fermion operators
    :param coefficients: list of VQE coefficients
    :param pool: pool of qubit operators
    :param simulator: simulation object
    :return:
    """

    # qubit hamiltonian to sparse (Jordan-Wigner transform is used)
    ansatz_qubit = transform_to_scaled_qubit(ansatz, coefficients)

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    if simulator._test_only:
        print('Non-Zero Gradients (Exact circuit evaluation)')
    else:
        print('Non-Zero Gradients (Simulated with {} shots)'.format(simulator._shots))

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(pool):

        # convert to qubit if necessary
        if not isinstance(operator, QubitOperator):
            operator = fermion_to_qubit(operator)

        # get gradient as dexp(c * A) / dc = < psi | [H, A] | psi >.
        commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian

        sampled_gradient = simulator.get_state_evaluation(commutator_hamiltonian, state_preparation_gates)

        # set absolute value for gradient
        sampled_gradient = np.abs(sampled_gradient)
        gradient_vector.append(sampled_gradient)

        if sampled_gradient > 1e-6:
            print("Operator {}: {:.6f}".format(i, sampled_gradient))

            # just for testing
            # print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient)

    return gradient_vector
