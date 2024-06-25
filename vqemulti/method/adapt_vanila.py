from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector
from vqemulti.gradient.simulation import simulate_gradient
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.errors import Converged
from vqemulti.method import zero_valued_coefficient_adaptvanilla, energy_worsening
import numpy as np
class AdapVanilla(Method):

    def __init__(self, energy_threshold, gradient_threshold, operator_update_number, operator_update_max_grad,
                 gradient_simulator, diff_threshold, hf_reference_fock, hamiltonian, ansatz, coefficients,
                  operators_pool, variance, iterations, energy_simulator):
        super().__init__(hf_reference_fock, hamiltonian, ansatz, coefficients,
                  operators_pool, variance, iterations, energy_simulator)
        self.energy_threshold = energy_threshold
        self.gradient_threshold = gradient_threshold
        self.operator_update_number = operator_update_number
        self.operator_update_max_grad = operator_update_max_grad
        self.gradient_simulator = gradient_simulator
        self.max_indices = 0
        self.diff_threshold = diff_threshold


    def update_ansatz(self):
        if self.gradient_simulator is None:
            gradient_vector = compute_gradient_vector(self.reference_hf,
                                                      self.hamiltonian,
                                                      self.ansatz,
                                                      self.coefficients,
                                                      self.operators_pool)
        else:
            self.gradient_simulator.update_model(precision=self.energy_threshold,
                                            variance=self.variance,
                                            n_coefficients=len(self.coefficients),
                                            n_qubits=self.hamiltonian.n_qubits)

            gradient_vector = simulate_gradient(self.reference_hf,
                                                self.hamiltonian,
                                                self.ansatz,
                                                self.coefficients,
                                                self.operators_pool,
                                                self.gradient_simulator)

        total_norm = np.linalg.norm(gradient_vector)

        print("\nTotal gradient norm: {:12.6f}".format(total_norm))

        if total_norm < self.gradient_threshold:
            if len(self.iterations['energies']) > 0:
                energy = self.iterations['energies'][-1]
            else:
                energy = get_adapt_vqe_energy(self.coefficients, self.ansatz, self.reference_hf, self.hamiltonian,
                                              self.energy_simulator)

            raise Converged(message='Converge archived due to gradient norm threshold', energy=energy,
                            ansatz=self.ansatz, indices=self.ansatz.get_index(self.operators_pool),
                            coefficients= self.coefficients, iterations=self.iterations, variance=self.variance)

        # primary selection of operators
        max_indices = np.argsort(gradient_vector)[-self.operator_update_number:][::-1]
        self.max_indices = max_indices                                                       #CHECK WITH ABEL!!!!!
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
        repeat_operator = len(max_indices) == len(self.ansatz.get_index(self.operators_pool)[-len(max_indices):]) and \
                          np.all(
                              np.array(max_indices) == np.array(self.ansatz.get_index(self.operators_pool)[-len(max_indices):]))

        # if repeat operator finish adaptVQE
        if repeat_operator:
            raise Converged(message='Converge archived due to repeated operator', energy=self.iterations['energies'][-1],
                            ansatz=self.ansatz, indices=self.ansatz.get_index(self.operators_pool), coefficients=self.coefficients,
                            iterations=self.iterations, variance=self.variance)

        # Initialize the coefficient of the operator that will be newly added at 0
        for max_index, max_operator in zip(max_indices, max_operators):
            self.coefficients.append(0)
            self.ansatz.append(max_operator)

        return self.ansatz, self.coefficients

    def check_convergence(self, params_check_convergence):
        params_check_convergence['max_indices'] = self.max_indices
        params_check_convergence['diff_threshold'] = self.diff_threshold
        criteria = [zero_valued_coefficient_adaptvanilla, energy_worsening]
        super().check_covergence(criteria, params_check_convergence)