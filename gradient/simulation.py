import openfermion

from utils import convert_hamiltonian, group_hamiltonian, transform_to_scaled_qubit
from gradient.exact import prepare_adapt_state
from energy.simulation.tools_penny import measure_expectation, get_exact_state_evaluation, build_gradient_ansatz
from energy.simulation import get_preparation_gates, get_preparation_gates_trotter, get_sampled_state_evaluation
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import numpy as np


def print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient):
    # compute exact gradient using matrix representation for comparison
    # qubit to sparse using Jordan-Wigner transform
    from gradient.exact import calculate_gradient
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


def simulate_gradient(hf_reference_fock, qubit_hamiltonian, ansatz, coefficients, pool, shots=10000,
                      trotter=True, trotter_steps=1, test_only=True):
    """
    simulate gradient using quantum computer simulators (Cirq, pennylane)

    :param hf_reference_fock:  reference HF state in Fock space vector
    :param qubit_hamiltonian: hamiltonian in qubit operators
    :param ansatz: VQE ansatz in qubit/Fermion operators
    :param coefficients: list of VQE coefficients
    :param pool: pool of qubit operators
    :param shots: number of samples
    :param test_only: If True get exact expectation gradient (test circuit)
    :return:
    """

    # qubit hamiltonian to sparse (Jordan-Wigner transform is used)
    ansatz_qubit = transform_to_scaled_qubit(ansatz, coefficients)

    if trotter:
        state_preparation_gates = get_preparation_gates_trotter(ansatz_qubit,
                                                                trotter_steps,
                                                                hf_reference_fock)

    else:
        state_preparation_gates = get_preparation_gates(ansatz_qubit,
                                                        hf_reference_fock)

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(pool):

        print("Operator {}".format(i))

        # convert to qubit if necessary
        if not isinstance(operator, openfermion.QubitOperator):
            operator = openfermion.jordan_wigner(operator)

        # get gradient as dexp(c * A) / dc = < psi | [H, A] | psi >.
        commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian

        if test_only:
            # Calculate the exact gradient in this state using simulator
            sampled_gradient = get_exact_state_evaluation(commutator_hamiltonian, state_preparation_gates).real

        else:
            # Sample the gradient in this state using simulator
            sampled_gradient = get_sampled_state_evaluation(commutator_hamiltonian, state_preparation_gates, shots)

        # set absolute value for gradient
        sampled_gradient = np.abs(sampled_gradient)
        gradient_vector.append(sampled_gradient)

        if test_only:
            print('Exact gradient (simulator): {:.6f}'.format(sampled_gradient))
        else:
            print("Simulated gradient: {:.6f} ({} shots)".format(sampled_gradient, shots))

        # just for testing (to be removed)
        print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient)

    return gradient_vector
