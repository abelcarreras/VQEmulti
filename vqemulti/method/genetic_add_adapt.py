from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
from copy import deepcopy
import numpy as np



class Genetic_Add_Adapt(Method):

    def __init__(self, gradient_threshold, diff_threshold, coeff_tolerance, gradient_simulator, beta, alpha):

        self.gradient_threshold = gradient_threshold
        self.diff_threshold = diff_threshold
        self.coeff_tolerance = coeff_tolerance
        self.gradient_simulator = gradient_simulator
        self.beta = beta
        self.alpha = alpha

        # Convergence criteria definition for this method
        self.criteria_list = [zero_valued_coefficient_adaptvanilla]
        self.params_convergence = {'coeff_tolerance': self.coeff_tolerance, 'diff_threshold': self.diff_threshold,
                                   }



    def update_ansatz(self, ansatz, coefficients, iterations):

        # Select the mutation that is going to happen
        if self.gradient_simulator is None:
            gradient_vector = compute_gradient_vector(self.reference_hf,
                                                      self.hamiltonian,
                                                      ansatz,
                                                      coefficients,
                                                      self.operators_pool)
        else:
            self.gradient_simulator.update_model(precision=self.energy_threshold,
                                            variance=self.variance,
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

        if len(coefficients)>2 and abs(iterations['energies'][-2]-iterations['energies'][-1])<self.alpha:
            add_probs = []
            for gradient in gradient_vector:
                prob_distribution = (2 * np.arctan(abs(gradient)*self.beta))/np.pi
                add_probs.append(prob_distribution)
            # Normalize add probability
            normalized_add_probs = np.array(add_probs)/np.sum(add_probs)

            # Create wheel for selecting the new ansatz
            def makeWheel(normalized_add_probs):
                wheel = []
                init_point = 0
                for i in range(len(normalized_add_probs)):
                    f = normalized_add_probs[i]
                    wheel.append((init_point, init_point + f, i))
                    init_point += f
                return wheel

            wheel = makeWheel(normalized_add_probs)
            print('wheel', wheel)
            # Here we generate the random position of the first pointer
            r = np.random.rand()
            for j in range(len(wheel)):
                if wheel[j][0] <= r < wheel[j][1]:
                    selected = wheel[j][2]
            new_operator = np.array(self.operators_pool)[selected]

            print("Selected: {} (norm {:.6f})".format(selected, gradient_vector[selected]))

            # check if repeated operator
            if len(coefficients)>0:
                repeat_operator = np.all(np.array(selected) == np.array(ansatz.get_index(self.operators_pool)[-1:]))
                # if repeat operator finish adaptVQE
                if repeat_operator:
                    raise Converged(message='Converge archived due to repeated operator')


            coefficients.append(0)
            ansatz.append(new_operator)

        else:
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

    def second_oprtunity(self, ansatz, coefficients):
        print('GIVING IT A SECOND OPRTUNITY')
        indeces = ansatz.get_index(self.operators_pool)
        ansatz = ansatz[:-1]
        coefficients = coefficients[:-1]
        if self.gradient_simulator is None:
            gradient_vector = compute_gradient_vector(self.reference_hf,
                                                      self.hamiltonian,
                                                      ansatz,
                                                      coefficients,
                                                      self.operators_pool)
        else:
            self.gradient_simulator.update_model(precision=self.energy_threshold,
                                            variance=self.variance,
                                            n_coefficients=len(coefficients),
                                            n_qubits=self.hamiltonian.n_qubits)

            gradient_vector = simulate_gradient(self.reference_hf,
                                                self.hamiltonian,
                                                ansatz,
                                                coefficients,
                                                self.operators_pool,
                                                self.gradient_simulator)
        gradient_vector[indeces[-1]] = 0
        total_norm = np.linalg.norm(gradient_vector)
        print("\nTotal gradient norm: {:12.6f}".format(total_norm))
        print('THE GRADIENT VECTOR IS', gradient_vector)

        if total_norm < self.gradient_threshold:
            raise Converged(message='Converge archived due to gradient norm threshold')

        if len(coefficients) > 2 and abs(self.iterations['energies'][-2] - self.iterations['energies'][-1]) < self.alpha:
            add_probs = []
            for gradient in gradient_vector:
                prob_distribution = (2 * np.arctan(abs(gradient) * self.beta)) / np.pi
                add_probs.append(prob_distribution)
            # Normalize add probability
            normalized_add_probs = np.array(add_probs) / np.sum(add_probs)

            # Create wheel for selecting the new ansatz
            def makeWheel(normalized_add_probs):
                wheel = []
                init_point = 0
                for i in range(len(normalized_add_probs)):
                    f = normalized_add_probs[i]
                    wheel.append((init_point, init_point + f, i))
                    init_point += f
                return wheel

            wheel = makeWheel(normalized_add_probs)
            print('wheel', wheel)
            # Here we generate the random position of the first pointer
            r = np.random.rand()
            for j in range(len(wheel)):
                if wheel[j][0] <= r < wheel[j][1]:
                    selected = wheel[j][2]
            new_operator = np.array(self.operators_pool)[selected]

            print("Selected: {} (norm {:.6f})".format(selected, gradient_vector[selected]))

            # check if repeated operator
            if len(coefficients) > 0:
                repeat_operator = np.all(np.array(selected) == np.array(ansatz.get_index(self.operators_pool)[-1:]))
                # if repeat operator finish adaptVQE
                if repeat_operator:
                    raise Converged(message='Converge archived due to repeated operator')

            coefficients.append(0)
            ansatz.append(new_operator)

        else:
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

