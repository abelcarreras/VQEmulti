import openfermion

from utils import convert_hamiltonian, group_hamiltonian, transform_to_qubit
from gradient.exact import prepare_adapt_state
from energy.simulation.tools_penny import measure_expectation, get_exact_state_evaluation, build_gradient_ansatz
from energy.simulation import get_preparation_gates, get_preparation_gates_trotter
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
import numpy as np
import scipy


def get_sampled_gradient(qubitOperator, qubitHamiltonian, statePreparationGates, shots=1000):
    """
    Given an operator A, samples the gradient of the energy with respect to the coefficient c of
    the operator exp(c * A), at c = 0, in a given state.
    Uses dexp(c*A)/dc = <psi|[H,A]|psi>.

    :param qubitOperator: qubit operator (A) respect to which the gradient is computed
    :param qubitHamiltonian: hamiltonian in qubits operators
    :param statePreparationGates: list of gates used to prepare the state
    :param n_qubits: number of qubits
    :param shots: number of samples
    :return: the sampled expectation value of <psi|[H,A]|psi>, that is an estimate of the value of the gradient
    """

    # get number of qubits
    n_qubits = count_qubits(qubitHamiltonian)

    # group commutator hamiltonian for more efficient quantum computer computation
    commutator_hamiltonian = qubitHamiltonian * qubitOperator - qubitOperator * qubitHamiltonian
    formatted_commutator = convert_hamiltonian(commutator_hamiltonian)
    grouped_commutator = group_hamiltonian(formatted_commutator)

    # Obtain the experimental expectation value for each Pauli string by
    # calling the measureExpectation function, and perform the necessary weighed
    # sum to obtain the energy expectation value
    commutator = 0
    for main_string, sub_hamiltonian in grouped_commutator.items():
        expectation_value = measure_expectation(main_string,
                                                sub_hamiltonian,
                                                shots,
                                                statePreparationGates,
                                                n_qubits)
        commutator += expectation_value

    assert commutator.imag < 1e-5

    return commutator.real



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
    ansatz_qubit, coefficients = transform_to_qubit(ansatz, coefficients)

    if trotter:
        state_preparation_gates = get_preparation_gates_trotter(coefficients,
                                                                ansatz_qubit,
                                                                trotter_steps,
                                                                hf_reference_fock)

    else:
        state_preparation_gates = get_preparation_gates(coefficients,
                                                        ansatz_qubit,
                                                        hf_reference_fock)

    # Calculate and print gradients
    gradient_vector = []
    for i, operator in enumerate(pool):

        print("Operator {}".format(i))

        if isinstance(operator, openfermion.FermionOperator):
            operator = openfermion.jordan_wigner(operator)

        if not test_only:
            sampled_gradient = np.abs(get_sampled_gradient(operator,
                                                           qubit_hamiltonian,
                                                           state_preparation_gates,
                                                           shots=shots))
        else:
            # Calculate the exact energy in this state
            commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian
            sampled_gradient = np.abs(get_exact_state_evaluation(commutator_hamiltonian, state_preparation_gates).real)

        if test_only:
            print('Exact gradient (simulator): {:.6f}'.format(sampled_gradient))
        else:
            print("Simulated gradient: {:.6f} ({} shots)".format(sampled_gradient, shots))

        gradient_vector.append(sampled_gradient)

        compute_exact_sparse = True
        if compute_exact_sparse:
            # compute exact gradient using matrix representation for comparison
            # qubit to sparse using Jordan-Wigner transform
            from gradient.exact import calculate_gradient
            sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian) # Get the current hamiltonian.
            sparse_state = prepare_adapt_state(hf_reference_fock, ansatz_qubit, coefficients) # Get the current state.
            sparse_operator = get_sparse_operator(operator, count_qubits(qubit_hamiltonian))
            calculated_gradient = calculate_gradient(sparse_operator, sparse_state, sparse_hamiltonian)
            print("Exact gradient: {:.6f}".format(calculated_gradient))

            error = np.abs(sampled_gradient - calculated_gradient)

            if abs(calculated_gradient) > 1e-3:
                print("Error: {0:.3f} ({1:.3f}%)\n".format(error, 100 * error / calculated_gradient))
            else:
                print("Error: {0:.3f} NA%)\n".format(error))


    return gradient_vector
