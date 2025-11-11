from openfermion.ops import FermionOperator
from openfermion.transforms import bravyi_kitaev
from openfermion import get_fermion_operator
from openfermion import QubitOperator

import numpy as np


def fock_to_bk_interleaved(fock_vector):
    """
    Compute Bravyi-Kitaev transformed occupation vector from Fock space occupation vector.
    Assumes interleaved ordering: [α₀, β₀, α₁, β₁, ...]
    """
    n = len(fock_vector)
    bk_vector = np.zeros(n, dtype=int)

    for i in range(n):
        # For each i, compute the parity set P(i)
        parity = 0
        k = i
        while k >= 0:
            # Add fock[k] if k is in the update set of i (based on binary representation)
            if (k & i) == k:
                parity ^= fock_vector[k]
            k -= 1
        bk_vector[i] = parity

    return bk_vector

fock_vector = [1, 1, 1, 1, 0, 0, 0, 0]
bk_vector = fock_to_bk_interleaved(fock_vector)
print(bk_vector)