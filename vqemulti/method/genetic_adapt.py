from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
from copy import deepcopy
import numpy as np



class GeneticAdapt(Method):

    def __init__(self, gradient_threshold,
                 diff_threshold,
                 coeff_tolerance,
                 gradient_simulator,
                 beta,
                 min_iterations=0):

        self.gradient_threshold = gradient_threshold
        self.diff_threshold = diff_threshold
        self.coeff_tolerance = coeff_tolerance
        self.gradient_simulator = gradient_simulator
        self.beta = beta

        # Convergence criteria definition for this method
        self.criteria_list = [zero_valued_coefficient_adaptvanilla]
        self.params_convergence = {'coeff_tolerance': self.coeff_tolerance, 'diff_threshold': self.diff_threshold,
                                   'min_iterations': min_iterations}



    def update_ansatz(self, ansatz, iterations):
        coefficients = deepcopy(iterations['coefficients'][-1])
        # Select the mutation that is going to happen
        # Create delete probabilities
        delete_probs = []
        for coeff in coefficients:
            prob_distribution = np.exp(-abs(coeff)*self.beta)
            delete_probs.append(prob_distribution/(len(coefficients)))
        # Normalize delete probability
        normalized_delete_probs = np.array(delete_probs)
        delete_probs = normalized_delete_probs.tolist()
        add_probability = 1 - np.sum(delete_probs)
        delete_probs.append(add_probability)
        all_probs = deepcopy(delete_probs)
        # Create wheel for selecting the new ansatz
        def makeWheel(all_probs):
            wheel = []
            init_point = 0
            for i in range(len(all_probs)):
                f = all_probs[i]
                wheel.append((init_point, init_point + f, i))
                init_point += f
            return wheel

        wheel = makeWheel(all_probs)
        # Here we generate the random position of the first pointer
        r = np.random.rand()
        for j in range(len(wheel)):
            if wheel[j][0] <= r < wheel[j][1]:
                selected = wheel[j][2]

        # Perform the selected action
        if selected == len(coefficients) or len(coefficients) < 2:
            # Let's generate the add peasant
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

            # primary selection of operators
            max_indices = np.argsort(gradient_vector)[-1:][::-1]

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
            if repeat_operator:
                raise Converged(message='Converge archived due to repeated operator')

            # Initialize the coefficient of the operator that will be newly added at 0
            for max_index, max_operator in zip(max_indices, max_operators):
                coefficients.append(0)
                ansatz.append(max_operator)

        else:
            first_half_ansatz = ansatz[0:selected]
            second_half_ansatz = ansatz[selected+1:]
            for i in range(len(second_half_ansatz._list)):
                first_half_ansatz.append(second_half_ansatz._list[i])
            ansatz = first_half_ansatz
            first_half_coeffs = coefficients[0:selected]
            second_half_coeffs = coefficients[selected+1:]
            coefficients = first_half_coeffs + second_half_coeffs





        return ansatz, coefficients

