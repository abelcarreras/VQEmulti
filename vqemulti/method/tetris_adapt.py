from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector
from vqemulti.gradient.simulation import simulate_gradient
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
from copy import deepcopy
import numpy as np


def qubits_operator_action(pool, indeces):
    """Takes a list of indices of operators and return a set of integers corresponding to qubit indices the qubit
    operators act on.

    Returns:
        list: Set of qubit indices.
    """
    qubit_indices = set()
    if len(indeces) == 1:
        index = int(indeces[0])
        for term in pool[index].terms:
            if term:
                indices = list(zip(*term))[0]
                qubit_indices.update(indices)
    else:
        for i in range(len(indeces)):
            for term in pool[indeces[i]].terms:
                if term:
                    indices = list(zip(*term))[0]
                    qubit_indices.update(indices)
    return qubit_indices



class AdapTetris(Method):

    def __init__(self, gradient_threshold, diff_threshold, coeff_tolerance, gradient_simulator, operator_update_max_grad
                ):

        self.gradient_threshold = gradient_threshold
        self.diff_threshold = diff_threshold
        self.coeff_tolerance = coeff_tolerance
        self.gradient_simulator = gradient_simulator
        self.operator_update_max_grad = operator_update_max_grad

        # Convergence criteria definition for this method
        self.criteria_list = [zero_valued_coefficient_adaptvanilla, energy_worsening]
        self.params_convergence = {'coeff_tolerance': self.coeff_tolerance, 'diff_threshold': self.diff_threshold}



    def update_ansatz(self, ansatz, iterations):
        coefficients = deepcopy(iterations['coefficients'][-1])
        if self.gradient_simulator is None:
            gradient_vector = compute_gradient_vector(self.reference_hf,
                                                      self.hamiltonian,
                                                      ansatz,
                                                      coefficients,
                                                      self.operators_pool)
        else:
            self.gradient_simulator.update_model(precision=self.energy_threshold,
                                            variance=iterations['variance'][-1],
                                            n_coefficients=len(coefficients),
                                            n_qubits=self.hamiltonian.n_qubits)

            gradient_vector = simulate_gradient(self.reference_hf,
                                                self.hamiltonian,
                                                ansatz,
                                                coefficients,
                                                self.operators_pool,
                                                self.gradient_simulator)

        total_norm = np.linalg.norm(gradient_vector)

        print("\nTotal gradient norm: {:12.6f}".format(total_norm))

        if total_norm < self.gradient_threshold:
            raise Converged(message='Converge archived due to gradient norm threshold')

        # Order the indices depending on the size of the gradient
        indices_sorted = np.argsort(gradient_vector)[::-1]

        # Start adding the operator with the largest gradient
        max_indices = np.argsort(gradient_vector)[-1:][::-1]

        # Then add the rest of operators
        for i in range(len(indices_sorted)-1):
            index_evaluated_op = [indices_sorted[i+1].tolist()]
            if (qubits_operator_action(self.operators_pool, index_evaluated_op) & qubits_operator_action(self.operators_pool, max_indices)):
                pass
            else:
                max_indices = np.append(max_indices, indices_sorted[i+1])

        # Delete operator with too small gradient
        while True:
            max_gradients = np.array(gradient_vector)[max_indices]
            max_dev = np.max(np.std(max_gradients))
            if max_dev/np.max(max_gradients) > self.operator_update_max_grad:
                max_indices = max_indices[:-1]
            else:
                break

        # get gradients/operators update list
        max_gradients = np.array(gradient_vector)[max_indices]
        max_operators = np.array(self.operators_pool)[max_indices]

        for max_index, max_gradient in zip(max_indices, max_gradients):
            print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

        # check if repeated operator
        repeat_operator = len(max_indices) == len(ansatz.get_index(self.operators_pool)[-len(max_indices):]) and \
                          np.all(
                              np.array(max_indices) == np.array(
                                  ansatz.get_index(self.operators_pool)[-len(max_indices):]))

        # if repeat operator finish adaptVQE
        if repeat_operator:
            raise Converged(message='Converge archived due to repeated operator')

        # Initialize the coefficient of the operator that will be newly added at 0
        for max_index, max_operator in zip(max_indices, max_operators):
            coefficients.append(0)
            ansatz.append(max_operator)

        return ansatz, coefficients




