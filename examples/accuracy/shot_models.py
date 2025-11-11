import numpy as np


# shots handling models
def get_evaluation_shots_model_simple(dict_parameters):

    energy_threshold = dict_parameters['precision']
    variance = dict_parameters['variance']

    n_shots = int(variance / energy_threshold ** 2)
    n_shots = max(1, n_shots)

    return energy_threshold, n_shots


def get_evaluation_shots_model_2(dict_parameters):
    energy_threshold = dict_parameters['precision']
    variance = dict_parameters['variance']
    n_coefficients = dict_parameters['n_coefficients']

    n_shots_dev = variance * n_coefficients ** (2 / 3) / (energy_threshold ** 2)

    return energy_threshold * 0.8, int(n_shots_dev)


def get_evaluation_shots_model_3(dict_parameters):
    s_prec = 1
    energy_threshold = dict_parameters['precision'] / s_prec
    variance = dict_parameters['variance']
    n_coefficients = dict_parameters['n_coefficients']

    n_shots_ave = variance * n_coefficients ** 2 * np.pi ** 2 / (100 * energy_threshold ** 2)
    n_shots_dev = variance * n_coefficients ** (2 / 3) / (energy_threshold ** 2)

    n_shots = int(max(n_shots_ave, n_shots_dev))

    tolerance = energy_threshold * min(1 / n_coefficients ** (1 / 3), 10 / (np.pi * n_coefficients))

    return tolerance, n_shots
