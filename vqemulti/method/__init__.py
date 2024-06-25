from vqemulti.errors import Converged


def zero_valued_coefficient_adaptvanilla(params_check_convergence, iterations, ansatz, operators_pool, variance):
    if abs(params_check_convergence['results_optimization'].x[-1]) < params_check_convergence['coeff_tolerance']:
        n_operators = len(params_check_convergence['max_indices'])
        raise Converged(message='Converge archived due to zero valued coefficient', energy=params_check_convergence['results_optimization'].fun,
                        ansatz=ansatz[:-n_operators], indices=ansatz.get_index(operators_pool)[:-n_operators],
                        coefficients=list(params_check_convergence['results_optimization'].x)[:-n_operators],
                        iterations=iterations, variance=variance)

def energy_worsening(params_check_convergence, iterations, ansatz, operators_pool, variance):
    if len(iterations['energies']) > 0 and iterations['energies'][-1] - params_check_convergence['results_optimization'].fun < params_check_convergence['diff_threshold']:
        n_operators = len(params_check_convergence['max_indices'])
        raise Converged(message='Converge archived due to not energy improvement',
                        energy=params_check_convergence['results_optimization'].fun,
                        ansatz=ansatz[:-n_operators], indices=ansatz.get_index(operators_pool)[:-n_operators],
                        coefficients=list(params_check_convergence['results_optimization'].x)[:-n_operators],
                        iterations=iterations, variance=variance)


class Method():
    def __init__(self, hf_reference_fock, hamiltonian, ansatz, coefficients,
                  operators_pool, variance, iterations, energy_simulator):
        self.reference_hf = hf_reference_fock
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.coefficients = coefficients
        self.operators_pool = operators_pool
        self.variance = variance
        self.iterations = iterations
        self.energy_simulator = energy_simulator

    def check_covergence(self, criteria_list, params_check_convergence):
        try:
            for criteria in criteria_list:
                criteria(params_check_convergence, self.iterations, self.ansatz, self.operators_pool, self.variance)
        except Converged as c:
            print(c.message)
            return {'energy': c.energy,
                    'ansatz': c.ansatz,
                    'indices': c.indices,
                    'coefficients': c.coefficients,
                    'iterations': c.iterations,
                    'variance': c.variance}

