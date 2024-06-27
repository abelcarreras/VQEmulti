from vqemulti.errors import Converged

def zero_valued_coefficient_adaptvanilla(params_convergence, iterations):
    if abs(iterations['coefficients'][-1][-1]) < params_convergence['coeff_tolerance']:
        raise Converged(message='Converge archived due to zero valued coefficient')

def energy_worsening(params_convergence, iterations):
    if len(iterations['energies']) > 1 and iterations['energies'][-2] - iterations['energies'][-1] < params_convergence['diff_threshold']:
        raise Converged(message='Converge archived due to not energy improvement')

