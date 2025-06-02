from vqemulti.pool.tools import OperatorList
from openfermion import QubitOperator
import numpy as np



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

            operators_list.append(term_aa)
            operators_list.append(term_bb)

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

                    term_aaaa = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ia, ja, aa, ba), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ia, ja, aa, ba), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ia, ja, aa, ba), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ia, ja, aa, ba), 0.125)

                    term_bbbb = QubitOperator('X{0} Y{1} X{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                QubitOperator('Y{0} X{1} X{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(ib, jb, ab, bb), 0.125) + \
                                QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                QubitOperator('X{0} X{1} Y{2} X{3}'.format(ib, jb, ab, bb), 0.125) - \
                                QubitOperator('X{0} X{1} X{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(ib, jb, ab, bb), 0.125) - \
                                QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(ib, jb, ab, bb), 0.125)

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

                    operators_list.append(term_aaaa)
                    operators_list.append(term_bbbb)
                    operators_list.append(term_abab)
                    operators_list.append(term_baba)
                    operators_list.append(term_abba)
                    operators_list.append(term_baab)

    return OperatorList(operators_list, antisymmetrize=False)


if __name__ == '__main__':
    pool = get_pool_qubit_sd(n_electrons=2, n_orbitals=4, frozen_core=0)
    print(pool)