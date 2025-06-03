from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
from vqemulti.energy import exact_adapt_vqe_energy
from openfermion import count_qubits
from copy import deepcopy
import numpy as np



class AdapVanilla(Method):

    def __init__(self,
                 gradient_threshold=1e-6,
                 diff_threshold=0,
                 coeff_tolerance=1e-10,
                 gradient_simulator=None,
                 operator_update_number=1,
                 operator_update_max_grad=2e-2,
                 min_iterations=0,
                 prune = False,
                 weight_coeffs = 2,
                 ops_account_for_thres = 4,
                 weight_position = 11):
        """
        :param gradient_threshold: total-gradient-norm convergence threshold (in Hartree)
        :param diff_threshold: missing description
        :param coeff_tolerance: Set upper limit value for coefficient to be considered as zero
        :param gradient_simulator: Simulator object used to obtain the gradient, if None do not use simulator (exact)
        :param operator_update_number: number of operators to add to the ansatz at each iteration
        :param operator_update_max_grad: max gradient relative deviation between operations that update together in one iteration
        """
        self.gradient_threshold = gradient_threshold
        self.diff_threshold = diff_threshold
        self.coeff_tolerance = coeff_tolerance
        self.gradient_simulator = gradient_simulator
        self.operator_update_number = operator_update_number
        self.operator_update_max_grad = operator_update_max_grad
        self.weight_coeffs = weight_coeffs
        self.weight_position = weight_position
        self.ops_account_for_thres = ops_account_for_thres
        self.prune = prune

        # Convergence criteria definition for this method
        if prune is not False:
            self.criteria_list = [zero_valued_coefficient_adaptvanilla]
        else:
            self.criteria_list = [zero_valued_coefficient_adaptvanilla, energy_worsening]
            #self.criteria_list = [ energy_worsening]

        self.params_convergence = {'coeff_tolerance': self.coeff_tolerance, 'diff_threshold': self.diff_threshold,
                                   'operator_update_number': self.operator_update_number, 'min_iterations': min_iterations}

    def update_ansatz(self, ansatz, iterations):

        coefficients = deepcopy(iterations['coefficients'][-1])
        n_qubits = count_qubits(self.hamiltonian)

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
                                                 n_qubits=n_qubits)

            gradient_vector = simulate_gradient(self.reference_hf,
                                                self.hamiltonian,
                                                ansatz,
                                                coefficients,
                                                self.operators_pool,
                                                self.gradient_simulator)

        total_norm = np.linalg.norm(gradient_vector)
        iterations['norms'].append(total_norm)

        print("\nTotal gradient norm: {:12.6f}".format(total_norm))

        if total_norm < self.gradient_threshold:
            raise Converged(message='Converge archived due to gradient norm threshold')

        # primary selection of operators
        max_indices = np.argsort(gradient_vector)[-self.operator_update_number:][::-1]

        # refine selection to ensure all operators are relevant
        while True:
            max_gradients = np.array(gradient_vector)[max_indices]
            max_dev = np.max(np.std(max_gradients))
            if max_dev / np.max(max_gradients) > self.operator_update_max_grad:
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
                              np.array(max_indices) == np.array(ansatz.get_index(self.operators_pool)[-len(max_indices):]))

        # if repeat operator finish adaptVQE
        repeat_operator = False
        if repeat_operator:
            raise Converged(message='Converge archived due to repeated operator')

        # Initialize the coefficient of the operator that will be newly added at 0
        for max_index, max_operator in zip(max_indices, max_operators):
            coefficients.append(0)
            ansatz.append(max_operator)

        return ansatz, coefficients


    def prune_ansatz(self, coefficients, ansatz, hf_reference_fock, hamiltonian, energy, operators_pool):
        if self.prune is not False:
            # Get the coefficients absolute value
            absolute_value_coeffs = [abs(coeff) for coeff in coefficients]

            # Calculate the decision factor for each operator
            decision_factor = []
            for i in range(len(absolute_value_coeffs)):
                position_contribution = np.exp(-self.weight_position * i / len(absolute_value_coeffs))
                coeff_contribution = 1 / (absolute_value_coeffs[i] ** self.weight_coeffs)
                decision_factor.append(position_contribution * coeff_contribution)
            decision_factor_normalized = []
            for i in range(len(decision_factor)):
                decision_factor_normalized.append(decision_factor[i] / np.sum(decision_factor))

            chosen_operator_factor = max(decision_factor_normalized)
            selected_operator_position = decision_factor_normalized.index(chosen_operator_factor)

            # Threshold calculation
            threshold = 0.01 * np.mean(absolute_value_coeffs[-self.ops_account_for_thres:])
            print('Threshold', threshold)
            if absolute_value_coeffs[selected_operator_position] < threshold:
                # Update the ansatz
                print('Selected operator position',
                      selected_operator_position, 'w/factor', decision_factor_normalized[selected_operator_position],
                      'w/index', ansatz.get_index(operators_pool)[selected_operator_position],
                      'and coeff', coefficients[selected_operator_position], ' IS REMOVED')
                print(ansatz[selected_operator_position])
                new_ansatz = ansatz[:selected_operator_position]
                for operator in ansatz[selected_operator_position + 1:]:
                    new_ansatz.append(operator)
                ansatz = new_ansatz
                coefficients.pop(selected_operator_position)
                energy = exact_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian)
                print('Coeffs', coefficients)

                return ansatz, coefficients, energy

            else:
                print('Selected operator positiion',
                      selected_operator_position, 'w/factor', decision_factor_normalized[selected_operator_position],
                      'and coeff', coefficients[selected_operator_position], 'not removed')
                print('Coeffs', coefficients)
                return ansatz, coefficients, energy
        else:
            return ansatz, coefficients, energy






