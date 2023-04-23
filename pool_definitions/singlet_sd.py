from openfermion import FermionOperator
from openfermion.utils import count_qubits, hermitian_conjugated
from openfermion.transforms import normal_ordered
import numpy as np


# start singlet_SQ

def get_singlet_sd(electronNumber, orbitalNumber):
    singlet_sd = []

    n_occ = int(np.ceil(electronNumber / 2))
    n_vir = orbitalNumber - n_occ
    #print(n_occ,n_vir, orbitalNumber)


    for i in range(0,n_occ):
        ia = 2*i
        ib = 2*i+1

        for a in range(0,n_vir):
            aa = 2*n_occ + 2*a
            ab = 2*n_occ + 2*a+1

            termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
            termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))

            termA -= hermitian_conjugated(termA)

            termA = normal_ordered(termA)

            #Normalize
            coeffA = 0
            for t in termA.terms:
                coeff_t = termA.terms[t]
                coeffA += coeff_t * coeff_t

            if termA.many_body_order() > 0:
                termA = termA/np.sqrt(coeffA)
                singlet_sd.append(termA)


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

                    termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                    termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                    termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))

                    termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                    termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                    termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                    termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)

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
                        singlet_sd.append(termA)

                    if termB.many_body_order() > 0:
                        termB = termB/np.sqrt(coeffB)
                        singlet_sd.append(termB)

    return singlet_sd
