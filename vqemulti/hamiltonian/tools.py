from vqemulti.utils import fermion_to_qubit, get_fermion_operator
from openfermion import FermionOperator
import numpy as np



def get_compressed_fermion_hamiltonian(hamiltonian, tolerance=1e-4):
    """
    Get compressed Fermionic Hamiltonian from a Hamiltonian object.
    All hamiltonian terms in FermionOperators are removed if smaller in magnitude than tolerance.

    :param hamiltonian: FermionOperator or InteractionOperator
    :param tolerance: tolerance value
    :return: FermionOperator
    """

    if not isinstance(hamiltonian, FermionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)

    hamiltonian = get_fermion_operator(hamiltonian)
    # print('H terms:', len(hamiltonian.terms))
    hamiltonian.compress(tolerance)
    # print('H terms compress:', len(hamiltonian.terms))
    return hamiltonian


def get_compressed_qubit_hamiltonian(hamiltonian, tolerance=1e-4):
    """
    Get compressed Qubit Hamiltonian from a Hamiltonian object.
    All hamiltonian terms in QubitOperators are removed if smaller in magnitude than tolerance.

    :param hamiltonian: QubitOperator, FermionOperator or InteractionOperator
    :param tolerance: tolerance value
    :return: QubitOperator
    """

    hamiltonian = fermion_to_qubit(hamiltonian)
    # print('H terms:', len(hamiltonian.terms))
    hamiltonian.compress(tolerance)
    # print('H terms compress:', len(hamiltonian.terms))
    return hamiltonian


def get_qdrift_hamiltonian(hamiltonian, n_drift_terms):
    """
    Get qdrift approximated hamiltonian from a Hamiltonian object.
    The returned operator do not implement the time but is already normalized

    :param hamiltonian: FermionOperator or InteractionOperator
    :param n_drift_terms: number of drift terms
    :return: QubitOperator
    """
    hamiltonian = fermion_to_qubit(hamiltonian)
    # print('H terms:', len(hamiltonian.terms))

    coeff_list = []
    op_list = []
    for op in hamiltonian.get_operators():

        # remove constant part
        if len(list(op.terms.keys())[0])==0:
            continue

        coeff_list.append(list(op.terms.values())[0])
        op_list.append(op)

    lmb = np.sum(np.abs(coeff_list))
    probabilities = np.abs(coeff_list)
    probabilities /= np.sum(probabilities)
    n_ops = len(op_list)

    list_qdrift = []
    for _ in range(n_drift_terms):
        idx = np.random.choice(n_ops, p=probabilities)

        op = op_list[idx]
        p = lmb/ abs(coeff_list[idx])

        list_qdrift.append(op * p /n_drift_terms)

    return sum(list_qdrift)
