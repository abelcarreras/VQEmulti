from copy import deepcopy
import numpy as np
from vqemulti.energy import exact_adapt_vqe_energy

class Prune_adapt():
    def __init__(self,
                 coeff_weight=2,
                 alpha = 11):
        self.coeff_weight = coeff_weight
        self.alpha = alpha

    def load_values(self, coefficients, hf_reference_fock, hamiltonian, energy):
        self.coefficients = deepcopy(coefficients)
        self.hf_reference_fock = hf_reference_fock
        self.hamiltonian = hamiltonian
        self.energy = energy


    def run_pruning(self, ansatz):
        # Get the coefficients absolute value
        absolute_value_coeffs = [abs(coeff) for coeff in self.coefficients]

        # Calculate the decision factor for each operator
        decision_factor = []
        for i in range(len(absolute_value_coeffs)):
            position_contribution = np.exp(-self.alpha * i / len(absolute_value_coeffs))
            coeff_contribution = 1 / (absolute_value_coeffs[i] ** self.coeff_weight)
            decision_factor.append(position_contribution * coeff_contribution)
        decision_factor_normalized = []
        for i in range(len(decision_factor)):
            decision_factor_normalized.append(decision_factor[i] / np.sum(decision_factor))

        chosen_operator_factor = max(decision_factor_normalized)
        selected_operator_position = decision_factor_normalized.index(chosen_operator_factor)

        # Threshold calculation
        threshold = 0.1 * np.mean(absolute_value_coeffs[-4:])

        if absolute_value_coeffs[selected_operator_position] < threshold:
            # Update the ansatz
            new_ansatz = ansatz[:selected_operator_position]
            for operator in ansatz[selected_operator_position + 1:]:
                new_ansatz.append(operator)
            ansatz = new_ansatz
            self.coefficients.pop(selected_operator_position)
            new_energy = exact_adapt_vqe_energy(self.coefficients, ansatz, self.hf_reference_fock,
                                                self.hamiltonian)
            print('Selected operator position',
                  selected_operator_position, 'w/factor', decision_factor_normalized[selected_operator_position],
                  'and coeff', self.coefficients[selected_operator_position], ' IS REMOVED')

            return new_energy, self.coefficients, ansatz

        else:
            print('Selected operator positiion',
                  selected_operator_position, 'w/factor', decision_factor_normalized[selected_operator_position],
                  'and coeff', self.coefficients[selected_operator_position], 'not removed')
            return self.energy, self.coefficients, ansatz





