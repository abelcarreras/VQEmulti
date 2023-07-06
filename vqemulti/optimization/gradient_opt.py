import numpy as np
import scipy
import copy as cp



def prepare_state(ansatz_sparsemat, ref, coefficients):
    """
            Prepare state:
            exp{A1}exp{A2}exp{A3}...exp{An}|ref>
            """
    new_state = ref * 1.0
    for k in reversed(range(0, len(coefficients))):
        new_state = scipy.sparse.linalg.expm_multiply((coefficients[k] * ansatz_sparsemat[k]), new_state)

    return new_state


def circuit_gradient(sparse_hamiltonian, ansatz_sparsemat, ref, coefficients):

    grad = []
    new_ket = prepare_state(ansatz_sparsemat, ref, coefficients)
    new_bra = new_ket.transpose().conj()

    hbra = new_bra.dot(sparse_hamiltonian)
    term = 0
    ket = cp.deepcopy(new_ket)
    grad = Recurse(coefficients, grad, hbra, ket, term, ansatz_sparsemat)

    return np.asarray(grad)




def Recurse(parameters, grad, hbra, ket, term, ansatz_sparsemat):
    if term == 0:
        hbra = hbra
        ket = ket
    else:
        hbra = (scipy.sparse.linalg.expm_multiply(-ansatz_sparsemat[term - 1] * parameters[term - 1],
                                                  hbra.transpose().conj())).transpose().conj()
        ket = scipy.sparse.linalg.expm_multiply(-ansatz_sparsemat[term - 1] * parameters[term - 1], ket)
    grad.append((2 * hbra.dot(ansatz_sparsemat[term]).dot(ket).toarray()[0][0].real))
    if term < len(parameters) - 1:
        term += 1
        Recurse(parameters, grad, hbra, ket, term, ansatz_sparsemat)
    return np.asarray(grad)
