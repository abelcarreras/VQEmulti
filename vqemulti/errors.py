class NotConvergedError(Exception):
    def __init__(self, results):
        self.results = results

    def __str__(self):
        n_steps = len(self.results)
        return 'Not converged in {} iterations\n Increase max_iterations'.format(n_steps)

