from functools import partial


class Method:

    def __init__(self):
        self.criteria_list = []
        self.params_convergence = []

    def initialize_general_variables(self, hamiltonian, operators_pool, energy_threshold):
        self.hamiltonian = hamiltonian
        self.operators_pool = operators_pool
        self.energy_threshold = energy_threshold

    def get_criteria_list_convergence(self):
        self.new_criteria_list = []
        for function in self.criteria_list:
            newfunction = partial(function, self.params_convergence)
            self.new_criteria_list.append(newfunction)
        return self.new_criteria_list








