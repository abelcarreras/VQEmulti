from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated
from openfermion.transforms import normal_ordered
from vqemulti.pool.tools import OperatorList
import numpy as np


def get_pool_singlet_gsd(n_orbitals, frozen_core=0):
    """
    get pool of generalized unitary fermion operators of single and double excitations

    :param n_electrons: number of electrons in occupied space
    :param n_orbitals: number of total molecular orbitals
    :param frozen_core: number of frozen orbitals
    :return: operators pool
    """
    n_orbitals = n_orbitals - frozen_core

    singlet_gsd = []

    for p in range(0, n_orbitals):
        pa = 2*p
        pb = 2*p+1

        for q in range(p, n_orbitals):
            qa = 2*q
            qb = 2*q+1

            termA =  FermionOperator(((pa,1),(qa,0)))
            termA += FermionOperator(((pb,1),(qb,0)))

            termA -= hermitian_conjugated(termA)
            termA = normal_ordered(termA)

            #Normalize
            coeffA = 0
            for t in termA.terms:
                coeff_t = termA.terms[t]
                coeffA += coeff_t * coeff_t

            if termA.many_body_order() > 0:
                termA = termA/np.sqrt(coeffA)
                singlet_gsd.append(termA)

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

                    termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                    termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                    termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                    termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                    termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                    termA -= hermitian_conjugated(termA)
                    termB -= hermitian_conjugated(termB)

                    termA = normal_ordered(termA)
                    termB = normal_ordered(termB)

                    #Normalize
                    coeffA = 0
                    coeffB = 0
                    for t in termA.terms:
                        coeff_t = termA.terms[t]
                        coeffA += coeff_t * coeff_t
                    for t in termB.terms:
                        coeff_t = termB.terms[t]
                        coeffB += coeff_t * coeff_t

                    if termA.many_body_order() > 0:
                        termA = termA/np.sqrt(coeffA)
                        singlet_gsd.append(termA)

                    if termB.many_body_order() > 0:
                        termB = termB/np.sqrt(coeffB)
                        singlet_gsd.append(termB)

    return OperatorList(singlet_gsd, antisymmetrize=False)

