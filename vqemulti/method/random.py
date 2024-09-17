from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
from copy import deepcopy
import numpy as np



class Random_Selection(Method):

    def __init__(self, gradient_threshold, diff_threshold, coeff_tolerance, gradient_simulator):

        self.gradient_threshold = gradient_threshold
        self.diff_threshold = diff_threshold
        self.coeff_tolerance = coeff_tolerance
        self.gradient_simulator = gradient_simulator
        # Convergence criteria definition for this method
        self.criteria_list = [zero_valued_coefficient_adaptvanilla]
        self.params_convergence = {'coeff_tolerance': self.coeff_tolerance, 'diff_threshold': self.diff_threshold,
                                   }


    def update_ansatz(self, ansatz, iterations):
        coefficients = deepcopy(iterations['coefficients'][-1])
        number_ops_pool = len(self.operators_pool)
        add_probs = []
        for i in range(number_ops_pool):
            prob_distribution = 1/number_ops_pool
            add_probs.append(prob_distribution)
        # Normalize add probability
        normalized_add_probs = np.array(add_probs)/np.sum(add_probs)
        print('PROBS', normalized_add_probs)
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
        # check if repeated operator

        coefficients.append(0)
        ansatz.append(new_operator)
        return ansatz, coefficients
