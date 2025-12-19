from vqemulti.method import Method
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
import numpy as np
import warnings


class AdapVanilla(Method):

    def __init__(self,
                 gradient_threshold=1e-6,
                 diff_threshold=0,
                 coeff_tolerance=1e-10,
                 gradient_simulator=None,
                 operator_update_number=1,
                 operator_update_max_grad=2e-2,
                 min_iterations=0):
        """
        :param gradient_threshold: total-gradient-norm convergence threshold (in Hartree)
        :param diff_threshold: missing description
        :param coeff_tolerance: Set upper limit value for coefficient to be considered as zero
        :param gradient_simulator: Simulator object used to obtain the gradient, if None do not use simulator (exact)
        :param operator_update_number: number of operators to add to the ansatz at each iteration
        :param operator_update_max_grad: max gradient relative deviation between operations that update together in one iteration
        :param min_iterations: force to do at least this number of iterations
        """
        self._gradient_threshold = gradient_threshold
        self._diff_threshold = diff_threshold
        self._coeff_tolerance = coeff_tolerance
        self._gradient_simulator = gradient_simulator
        self._operator_update_number = operator_update_number
        self._operator_update_max_grad = operator_update_max_grad

        # Convergence criteria definition for this method
        self.criteria_list = [zero_valued_coefficient_adaptvanilla, energy_worsening]
        self.params_convergence = {'coeff_tolerance': self._coeff_tolerance, 'diff_threshold': self._diff_threshold,
                                   'operator_update_number': self._operator_update_number, 'min_iterations': min_iterations}

    def update_ansatz(self, ansatz, iterations):

        if self._gradient_simulator is not None:
            self._gradient_simulator.update_model(precision=self.energy_threshold,
                                                  variance=iterations['variance'][-1],
                                                  n_coefficients=len(ansatz),
                                                  n_qubits=ansatz.n_qubits)

        gradient_vector = ansatz.pool_gradient_vector(self.hamiltonian, self.operators_pool, self._gradient_simulator)

        total_norm = np.linalg.norm(gradient_vector)
        iterations['norms'].append(total_norm)

        print("\nTotal gradient norm: {:12.6f}".format(total_norm))

        if total_norm < self._gradient_threshold:
            raise Converged(message='Converge archived due to gradient norm threshold')

        # primary selection of operators
        max_indices = np.argsort(gradient_vector)[-self._operator_update_number:][::-1]

        # refine selection to ensure all operators are relevant
        while True:
            max_gradients = np.array(gradient_vector)[max_indices]
            max_dev = np.max(np.std(max_gradients))
            if max_dev / np.max(max_gradients) > self._operator_update_max_grad:
                max_indices = max_indices[:-1]
            else:
                break

        # get gradients/operators update list
        max_gradients = np.array(gradient_vector)[max_indices]
        max_operators = np.array(self.operators_pool)[max_indices]

        for max_index, max_gradient in zip(max_indices, max_gradients):
            print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

        # check if repeated operator
        repeat_operator = len(max_indices) == len(ansatz.operators.get_index(self.operators_pool)[-len(max_indices):]) and \
                           np.all(np.array(max_indices) == np.array(ansatz.operators.get_index(self.operators_pool)[-len(max_indices):]))

        # if repeat operator finish adaptVQE
        if repeat_operator:
            warnings.warn('warning: repeated operator')
            return ansatz
            # raise Converged(message='Converge archived due to repeated operator')

        # Initialize the coefficient of the operator that will be newly added at 0
        for max_index, max_operator in zip(max_indices, max_operators):
            ansatz.add_operator(max_operator, 0.0)

        return ansatz






