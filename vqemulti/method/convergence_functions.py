from vqemulti.errors import Converged

def zero_valued_coefficient_adaptvanilla(params_check_convergence, iterations, ansatz, operators_pool, variance):
    print('PRIMERA')
    if abs(params_check_convergence['results_optimization'].x[-1]) < params_check_convergence['coeff_tolerance']:
        n_operators = params_check_convergence['operator_update_number']
        raise Converged(message='Converge archived due to zero valued coefficient', energy=params_check_convergence['results_optimization'].fun,
                        ansatz=ansatz[:-n_operators], indices=ansatz.get_index(operators_pool)[:-n_operators],
                        coefficients=list(params_check_convergence['results_optimization'].x)[:-n_operators],
                        iterations=iterations, variance=variance)

def energy_worsening(params_check_convergence, iterations, ansatz, operators_pool, variance):
    print('SEGUNDA')
    if len(iterations['energies']) > 0 and iterations['energies'][-1] - params_check_convergence['results_optimization'].fun < params_check_convergence['diff_threshold']:
        n_operators = params_check_convergence['operator_update_number']
        raise Converged(message='Converge archived due to not energy improvement',
                        energy=params_check_convergence['results_optimization'].fun,
                        ansatz=ansatz[:-n_operators], indices=ansatz.get_index(operators_pool)[:-n_operators],
                        coefficients=list(params_check_convergence['results_optimization'].x)[:-n_operators],
                        iterations=iterations, variance=variance)

