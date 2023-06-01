# spin_complement_gsd
from openfermion import FermionOperator
import openfermion


def get_pool_spin_complement_gsd(n_orbitals, frozen_core=0):
    """
    get pool of unitary fermion operators of spin complement single and double excitations

    :param n_orbitals: number of total molecular orbitals
    :return: operators pool
    """

    n_orbitals = n_orbitals - frozen_core

    spin_complement_gsd = []
    for p in range(0, n_orbitals):
        pa = 2*p
        pb = 2*p+1

        for q in range(p, n_orbitals):
            qa = 2*q
            qb = 2*q+1

            termA =  FermionOperator(((pa,1),(qa,0)))
            termA += FermionOperator(((pb,1),(qb,0)))

            termA -= openfermion.hermitian_conjugated(termA)

            termA = openfermion.normal_ordered(termA)

            if termA.many_body_order() > 0:
                spin_complement_gsd.append(termA)

    pq = -1
    for p in range(0, n_orbitals):
        pa = 2*p
        pb = 2*p+1

        for q in range(p, n_orbitals):
            qa = 2*q
            qb = 2*q+1

            pq += 1

            rs = -1
            for r in range(0, n_orbitals):
                ra = 2*r
                rb = 2*r+1

                for s in range(r, n_orbitals):
                    sa = 2*s
                    sb = 2*s+1

                    rs += 1

                    if(pq > rs):
                        continue

                    termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                    termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))

                    termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                    termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))

                    termC =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                    termC += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))

                    termA -= openfermion.hermitian_conjugated(termA)
                    termB -= openfermion.hermitian_conjugated(termB)
                    termC -= openfermion.hermitian_conjugated(termC)

                    termA = openfermion.normal_ordered(termA)
                    termB = openfermion.normal_ordered(termB)
                    termC = openfermion.normal_ordered(termC)

                    if termA.many_body_order() > 0:
                      spin_complement_gsd.append(termA)

                    if termB.many_body_order() > 0:
                        spin_complement_gsd.append(termB)

                    if termC.many_body_order() > 0:
                        spin_complement_gsd.append(termC)

    return spin_complement_gsd
