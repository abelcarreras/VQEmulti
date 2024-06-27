from vqemulti.method import Method
from vqemulti.gradient import compute_gradient_vector
from vqemulti.gradient.simulation import simulate_gradient
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.errors import Converged
from vqemulti.method.convergence_functions import zero_valued_coefficient_adaptvanilla, energy_worsening
import numpy as np


def operator_action(pool, index):
    """Return a set of integers corresponding to qubit indices the qubit
    operator acts on.

    Returns:
        list: Set of qubit indices.
    """

    qubit_indices = set()
    if len(index) == 1:
        index = int(index[0])
        for term in pool[index].terms:
            if term:
                indices = list(zip(*term))[0]
                qubit_indices.update(indices)

    else:
        for i in range(len(index)):
            for term in pool[index[i]].terms:
                if term:
                    indices = list(zip(*term))[0]
                    qubit_indices.update(indices)
    return qubit_indices



class AdapTetris(Method):

    pass

