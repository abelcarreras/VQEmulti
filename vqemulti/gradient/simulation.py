from vqemulti.utils import fermion_to_qubit
from vqemulti.gradient.exact import calculate_gradient, prepare_adapt_state
from openfermion import get_sparse_operator, QubitOperator, count_qubits
import numpy as np


def print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient):
    """
    print difference of gradient vector between exact anc calculated
    this function is designed for testing

    :param qubit_hamiltonian: hamiltonian in qubit operators
    :param hf_reference_fock: reference HF state in Fock space vector
    :param ansatz_qubit: VQE ansatz in qubit/Fermion operators
    :param operator: operator in qubits
    :param sampled_gradient: sampled gradient
    """

    sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)  # Get the current hamiltonian.
    sparse_state = prepare_adapt_state(hf_reference_fock, ansatz_qubit, [1] * len(ansatz_qubit))  # Get the current state.
    sparse_operator = get_sparse_operator(operator, count_qubits(qubit_hamiltonian))
    calculated_gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)
    print("Exact gradient: {:.6f}".format(calculated_gradient))

    error = np.abs(sampled_gradient - calculated_gradient)

    if abs(calculated_gradient) > 1e-3:
        print("Error: {0:.3f} ({1:.3f}%)\n".format(error, 100 * error / calculated_gradient))
    else:
        print("Error: {0:.3f} NA%)\n".format(error))


def simulate_gradient(hf_reference_fock, hamiltonian, ansatz, coefficients, pool, simulator):
    """
    Calculates the gradient of the energy with respect to adding new operator from a pool.
    To be used in adaptVQE poll gradients calculation  using simulator

    :param hf_reference_fock: reference HF state in Fock space vector
    :param hamiltonian: hamiltonian in qubit operators
    :param ansatz: VQE ansatz in qubit/Fermion operators
    :param coefficients: list of VQE coefficients
    :param pool: pool of qubit operators
    :param simulator: simulation object
    :return: the gradient_vector
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

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

        # set absolute value for gradient (sign is not important, only magnitude)
        sampled_gradient = np.abs(sampled_gradient)
        gradient_vector.append(sampled_gradient)

        if sampled_gradient > 1e-6:
            print("Operator {}: {:.6f}".format(i, sampled_gradient))

            # just for testing
            # print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient)

    return gradient_vector


def simulate_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Calculates the gradient of the energy with respect to the coefficients for VQE/adaptVQE Wave function using simulator.
    To be used as a gradient function in the energy minimization

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: gradient vector
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(ansatz):

        # convert to qubit if necessary
        if not isinstance(operator, QubitOperator):
            operator = fermion_to_qubit(operator)

        # get gradient as dexp(c * A) / dc = < psi | [H, A] | psi >.
        commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian

        sampled_gradient = simulator.get_state_evaluation(commutator_hamiltonian, state_preparation_gates)

        gradient_vector.append(sampled_gradient)

    #for g in gradient_vector:
    #    print('grad-: {:.8f}'.format(g))
    #print('-------')

    return gradient_vector

