from vqemulti.gradient import compute_gradient_vector
from vqemulti.gradient.simulation import simulate_gradient
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.errors import Converged
import numpy as np
def update_ansatz(hf_reference_fock, hamiltonian, ansatz, coefficients, gradient_simulator, energy_threshold,
                  operators_pool, variance, gradient_threshold, iterations, operator_update_number,
                  operator_update_max_grad, energy_simulator):
    if gradient_simulator is None:
        gradient_vector = compute_gradient_vector(hf_reference_fock,
                                                  hamiltonian,
                                                  ansatz,
                                                  coefficients,
                                                  operators_pool)
    else:
        gradient_simulator.update_model(precision=energy_threshold,
                                        variance=variance,
                                        n_coefficients=len(coefficients),
                                        n_qubits=hamiltonian.n_qubits)

        gradient_vector = simulate_gradient(hf_reference_fock,
                                            hamiltonian,
                                            ansatz,
                                            coefficients,
                                            operators_pool,
                                            gradient_simulator)

    total_norm = np.linalg.norm(gradient_vector)

    print("\nTotal gradient norm: {:12.6f}".format(total_norm))

    if total_norm < gradient_threshold:
        if len(iterations['energies']) > 0:
            energy = iterations['energies'][-1]
        else:
            energy = get_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, energy_simulator)

        raise Converged(message='Converge archived due to gradient norm threshold', energy=energy,
                        ansatz=ansatz, indices=ansatz.get_index(operators_pool), coefficients=coefficients,
                        iterations=iterations, variance=variance)

    # primary selection of operators
    max_indices = np.argsort(gradient_vector)[-operator_update_number:][::-1]

    # refine selection to ensure all operators are relevant
    while True:
        max_gradients = np.array(gradient_vector)[max_indices]
        max_dev = np.max(np.std(max_gradients))
        if max_dev /np.max(max_gradients) > operator_update_max_grad:
            max_indices = max_indices[:-1]
        else:
            break

    # get gradients/operators update list
    max_gradients = np.array(gradient_vector)[max_indices]
    max_operators = np.array(operators_pool)[max_indices]

    for max_index, max_gradient in zip(max_indices, max_gradients):
        print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

    # check if repeated operator
    repeat_operator = len(max_indices) == len(ansatz.get_index(operators_pool)[-len(max_indices):]) and \
                      np.all(np.array(max_indices) == np.array(ansatz.get_index(operators_pool)[-len(max_indices):]))

    # if repeat operator finish adaptVQE
    if repeat_operator:
        raise Converged(message='Converge archived due to repeated operator', energy = iterations['energies'][-1],
                        ansatz = ansatz, indices = ansatz.get_index(operators_pool), coefficients = coefficients,
                        iterations = iterations, variance = variance)


    # Initialize the coefficient of the operator that will be newly added at 0
    for max_index, max_operator in zip(max_indices, max_operators):
        coefficients.append(0)
        ansatz.append(max_operator)

    return ansatz, coefficients, max_indices
