
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
        self.criteria_list = []
        self.params_convergence = {}

    def get_criteria_list_convergence(self):
        return self.criteria_list, self.params_convergence






