from functools import partial
class Method():
    def __init__(self, hf_reference_fock, hamiltonian,
                 operators_pool, variance, iterations, energy_simulator):
        self.reference_hf = hf_reference_fock
        self.hamiltonian = hamiltonian
        self.operators_pool = operators_pool
        self.variance = variance
        self.iterations = iterations
        self.energy_simulator = energy_simulator
        self.criteria_list = []
        self.params_convergence = {}

    def get_criteria_list_convergence(self):
        self.new_criteria_list = []
        for function in self.criteria_list:
            newfunction = partial(function, self.params_convergence)
            self.new_criteria_list.append(newfunction)
        return self.new_criteria_list






