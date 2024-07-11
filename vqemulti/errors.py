class NotConvergedError(Exception):
    def __init__(self, results):
        self.results = results

    def __str__(self):
        n_steps = len(self.results['iterations'])
        return 'Not converged in {} iterations\n Increase max_iterations'.format(n_steps)


class Converged(Exception):
    def __init__(self, message):
        self.result_message = message

    @property
    def message(self):
        return self.result_message
