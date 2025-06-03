from vqemulti.pool.tools import OperatorList
from openfermion import QubitOperator
import numpy as np



def get_pool_qubit_gsd(n_orbitals, frozen_core=0):
    """
    get pool of generalized qubit operators of single and double excitations
    preserve Sz & N

    :param n_orbitals: number of total molecular orbitals
    :param frozen_core: number of frozen orbitals
    :return: operators pool
    """

    n_orbitals = n_orbitals - frozen_core

    operators_list = []

    for p in range(0, n_orbitals):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, n_orbitals):
            qa = 2 * q
            qb = 2 * q + 1

            if pa != qa:
                term_aa = QubitOperator('X{0} Y{1}'.format(pa, qa), 0.5) - \
                          QubitOperator('Y{0} X{1}'.format(pa, qa), 0.5)
                operators_list.append(1.j * term_aa)

            if pb != qb:
                term_bb = QubitOperator('X{0} Y{1}'.format(pb, qb), 0.5) - \
                          QubitOperator('Y{0} X{1}'.format(pb, qb), 0.5)
                operators_list.append(1.j * term_bb)

    operators_list_2e = []
    pq = -1
    for p in range(0, n_orbitals):
        pa = 2 * p
        pb = 2 * p + 1

        for q in range(p, n_orbitals):
            qa = 2 * q
            qb = 2 * q + 1

            pq += 1

            rs = -1
            for r in range(0, n_orbitals):
                ra = 2 * r
                rb = 2 * r + 1

                for s in range(r, n_orbitals):
                    sa = 2 * s
                    sb = 2 * s + 1

                    rs += 1

                    if (pq > rs):
                        continue

                    if len({pa, qa, ra, sa}) == 4:
                        term_aaaa = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pa, qa, ra, sa), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pa, qa, ra, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pa, qa, ra, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pa, qa, ra, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pa, qa, ra, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pa, qa, ra, sa), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pa, qa, ra, sa), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pa, qa, ra, sa), 0.125)
                        operators_list_2e.append(1.j * term_aaaa)

                    if len({pb, qb, rb, sb}) == 4:
                        term_bbbb = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pb, qb, rb, sb), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pb, qb, rb, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pb, qb, rb, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pb, qb, rb, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pb, qb, rb, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pb, qb, rb, sb), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pb, qb, rb, sb), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pb, qb, rb, sb), 0.125)
                        operators_list_2e.append(1.j * term_bbbb)

                    if pa != ra and qb != sb:
                        term_abab = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pa, qb, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pa, qb, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pa, qb, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pa, qb, ra, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pa, qb, ra, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pa, qb, ra, sb), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pa, qb, ra, sb), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pa, qb, ra, sb), 0.125)
                        operators_list_2e.append(1.j * term_abab)

                    if pb != rb and qa != sa:
                        term_baba = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pb, qa, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pb, qa, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pb, qa, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pb, qa, rb, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pb, qa, rb, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pb, qa, rb, sa), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pb, qa, rb, sa), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pb, qa, rb, sa), 0.125)
                        operators_list_2e.append(1.j * term_baba)

                    if pa != sa and qb != rb:
                        term_abba = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pa, qb, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pa, qb, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pa, qb, rb, sa), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pa, qb, rb, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pa, qb, rb, sa), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pa, qb, rb, sa), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pa, qb, rb, sa), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pa, qb, rb, sa), 0.125)
                        operators_list_2e.append(1.j * term_abba)

                    if pb != sb and qa != ra:
                        term_baab = QubitOperator('X{0} Y{1} X{2} X{3}'.format(pb, qa, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} X{1} X{2} X{3}'.format(pb, qa, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} Y{2} X{3}'.format(pb, qa, ra, sb), 0.125) + \
                                    QubitOperator('Y{0} Y{1} X{2} Y{3}'.format(pb, qa, ra, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} Y{2} X{3}'.format(pb, qa, ra, sb), 0.125) - \
                                    QubitOperator('X{0} X{1} X{2} Y{3}'.format(pb, qa, ra, sb), 0.125) - \
                                    QubitOperator('Y{0} X{1} Y{2} Y{3}'.format(pb, qa, ra, sb), 0.125) - \
                                    QubitOperator('X{0} Y{1} Y{2} Y{3}'.format(pb, qa, ra, sb), 0.125)
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
    pool = get_pool_qubit_gsd(n_orbitals=4, frozen_core=0)
    print(pool)