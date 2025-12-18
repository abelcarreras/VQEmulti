from openfermion import FermionOperator, hermitian_conjugated, normal_ordered
from vqemulti.pool.tools import OperatorList
from vqemulti.utils import fermion_to_qubit
import numpy as np


def get_ucc_generator(t1, t2, tolerance=1e-6, use_qubit=False, full_amplitudes=False):
    """
    Get unitary coupled cluster ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: define UCC ansatz in qubit operators
    :return: coefficients as list and ansatz as OperatorsList
    """

    operator_tot = FermionOperator()

    # 1-exciation terms
    if t1 is not None:
        n_spin_orbitals = len(t1)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                if abs(t1[i,j]) > tolerance:
                    operator = FermionOperator('{}^ {}'.format(i, j))
                    if not full_amplitudes:
                        operator_tot += t1[i, j] * operator - hermitian_conjugated(t1[i, j] * operator)
                    else:
                        operator_tot += t1[i, j] * operator

    # 2-excitation terms
    if t2 is not None:
        n_spin_orbitals = len(t2)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                for k in range(n_spin_orbitals):
                    for l in range(n_spin_orbitals):  # avoid duplicates
                        if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry
                            if abs(t2[i, j, k, l]) > tolerance:
                                operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                                if not full_amplitudes:
                                    operator_tot += 0.5 * t2[i, j, k, l] * operator - hermitian_conjugated(0.5 * t2[i, j, k, l] * operator)

                                else:
                                    operator_tot += 0.5 * t2[i, j, k, l] * operator

    assert normal_ordered(operator_tot + hermitian_conjugated(operator_tot)).isclose(FermionOperator.zero(), tolerance)

    operators = [operator_tot]
    coefficients = [1.0]

    if use_qubit:
        operators = [fermion_to_qubit(op) for op in operators]

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz
