from vqemulti.pool.tools import OperatorList
from openfermion import QubitOperator
import numpy as np


def _not_in_list(op, operator_list):
    return True
    for op1 in operator_list:
        if str(op1-op) == '0':
            return False
    return True


def get_pool_qubit_sd(n_electrons, n_orbitals, frozen_core=0):
    """
    get pool of qubit operators of single and double excitations
    preserve Sz & N

    :param n_electrons: number of electrons in occupied space
    :param n_orbitals: number of total molecular orbitals
    :param frozen_core: number of frozen orbitals
    :return: operators pool
    """
    operators_list = []

    n_occ = int(np.ceil(n_electrons / 2)) - frozen_core
    n_vir = n_orbitals - n_occ - frozen_core


    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for a in range(0,n_vir):
            aa = 2*n_occ + 2*a
            ab = 2*n_occ + 2*a+1

            term_aa = QubitOperator('X{0} Y{1}'.format(ia, aa), 0.5) - \
                      QubitOperator('Y{0} X{1}'.format(ia, aa), 0.5)

            term_bb = QubitOperator('X{0} Y{1}'.format(ib, ab), 0.5) - \
                      QubitOperator('Y{0} X{1}'.format(ib, ab), 0.5)

            operators_list.append(1.j * term_aa)
            operators_list.append(1.j * term_bb)

    operators_list_2e = []

    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for j in range(i,n_occ):
            ja = 2*j
            jb = 2*j+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1

                for b in range(a,n_vir):
                    ba = 2*n_occ + 2*b
                    bb = 2*n_occ + 2*b+1

                    if ia != ja and aa != ba:
                        term_aaaa = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(ia, ja, aa, ba), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ia, ja, aa, ba), 0.125)
                        operators_list_2e.append(1.j * term_aaaa)

                    if ib != jb and ab != bb:
                        term_bbbb = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(ib, jb, ab, bb), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ib, jb, ab, bb), 0.125)
                        operators_list_2e.append(1.j * term_bbbb)

                    term_abab = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ia, jb, aa, bb), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ia, jb, aa, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ia, jb, aa, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ia, jb, aa, bb), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ia, jb, aa, bb), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ia, jb, aa, bb), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ia, jb, aa, bb), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ia, jb, aa, bb), 0.125)

                    term_baba = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ib, ja, ab, ba), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ib, ja, ab, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ib, ja, ab, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ib, ja, ab, ba), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ib, ja, ab, ba), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ib, ja, ab, ba), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ib, ja, ab, ba), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ib, ja, ab, ba), 0.125)

                    term_abba = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ia, jb, ab, ba), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ia, jb, ab, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ia, jb, ab, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ia, jb, ab, ba), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ia, jb, ab, ba), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ia, jb, ab, ba), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ia, jb, ab, ba), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ia, jb, ab, ba), 0.125)

                    term_baab = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ib, ja, aa, bb), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ib, ja, aa, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ib, ja, aa, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ib, ja, aa, bb), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ib, ja, aa, bb), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ib, ja, aa, bb), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ib, ja, aa, bb), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ib, ja, aa, bb), 0.125)

                    operators_list_2e.append(1.j * term_abab)
                    operators_list_2e.append(1.j * term_baba)
                    operators_list_2e.append(1.j * term_abba)
                    operators_list_2e.append(1.j * term_baab)

    # TODO: improve this pool to not need to remove redundant operators
    # remove redundant operators
    unique_2e_list = []
    for i, op1 in enumerate(operators_list_2e):
        count = 0
        for op2 in unique_2e_list:
            if str(op1-op2) == '0':
                count += 1
        if count == 0:
            unique_2e_list.append(op1)

    operators_list = operators_list + unique_2e_list

    return OperatorList(operators_list, antisymmetrize=False)


if __name__ == '__main__':
    pool = get_pool_qubit_sd(n_electrons=2, n_orbitals=3, frozen_core=0)
    # print(pool)