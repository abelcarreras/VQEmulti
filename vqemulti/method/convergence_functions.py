from vqemulti.errors import Converged


def zero_valued_coefficient_adaptvanilla(params_convergence, iterations):
    if abs(iterations['coefficients'][-1][-1]) < params_convergence['coeff_tolerance']:
        min_iterations = params_convergence['min_iterations']
        if len(iterations['energies']) > min_iterations:
            raise Converged(message='Converge archived due to zero valued coefficient')


def energy_worsening(params_convergence, iterations):
    min_iterations = params_convergence['min_iterations']
    diff_threshold = params_convergence['diff_threshold']
    if (len(iterations['energies']) > min_iterations and
            iterations['energies'][-2] - iterations['energies'][-1] < diff_threshold):
        raise Converged(message='Converge archived due to not energy improvement')

