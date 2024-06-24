class NotConvergedError(Exception):
    def __init__(self, results):
        self.results = results

    def __str__(self):
        n_steps = len(self.results['iterations'])
        return 'Not converged in {} iterations\n Increase max_iterations'.format(n_steps)

class Converged(Exception):
    def __init__(self, message, energy, ansatz, indices, coefficients, iterations, variance):
        self.result_message = message
        self.result_energy = energy
        self.result_ansatz = ansatz
        self.result_indices = indices
        self.result_coefficients = coefficients
        self.result_iterations = iterations
        self.result_variance = variance

    @property
    def message(self):
        return self.result_message
    @property
    def energy(self):
        return self.result_energy
    @property
    def ansatz(self):
        return self.result_ansatz
    @property
    def indices(self):
        return self.result_indices
    @property
    def coefficients(self):
        return self.result_coefficients
    @property
    def iterations(self):
        return self.result_iterations
    @property
    def variance(self):
        return self.result_variance
