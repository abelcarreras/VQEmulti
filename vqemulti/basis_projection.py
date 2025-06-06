from vqemulti.pool.tools import OperatorList
from posym.basis import PrimitiveGaussian, BasisFunction
from openfermion import FermionOperator, QubitOperator, normal_ordered
from openfermion.utils import hermitian_conjugated
from vqemulti.utils import proper_order
from operator import mul
from functools import reduce
from pyscf import gto
import numpy as np


def get_basis_set_pyscf(pyscf_mol):
    """
    get basis functions from pyscf molecule object
    :param pyscf_mol: pyscf molecule
    :return: list of basis functions
    """

    basis_functions = []
    for bas_id in range(pyscf_mol.nbas):
        if False:
            print('coord', pyscf_mol.bas_coord(bas_id))
            print('angular', pyscf_mol.bas_angular(bas_id))
            print('atom', pyscf_mol.bas_atom(bas_id), pyscf_mol.atom_symbol(pyscf_mol.bas_atom(bas_id)))
            print('prim', pyscf_mol.bas_nprim(bas_id))
            print('contraction', pyscf_mol.bas_ctr_coeff(bas_id))
            print('exponents', pyscf_mol.bas_exp(bas_id))
            print('kappa', pyscf_mol.bas_kappa(bas_id))
            print('-------------------')

        bf_names = ['s', 'p', 'd', 'f']

        if pyscf_mol.bas_angular(bas_id) == 0:  # s
            primitives = []
            for prim_id, exponent in enumerate(pyscf_mol.bas_exp(bas_id)):
                primitives.append(PrimitiveGaussian(alpha=exponent))

            basis_functions.append(BasisFunction(primitives, pyscf_mol.bas_ctr_coeff(bas_id),
                                                 center=pyscf_mol.bas_coord(bas_id),
                                                 label='{}:{}'.format(pyscf_mol.atom_symbol(pyscf_mol.bas_atom(bas_id)),
                                                                      bf_names[pyscf_mol.bas_angular(bas_id)]))
                                   )

        elif pyscf_mol.bas_angular(bas_id) == 1:  # p
            for l_set in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
                primitives = []
                for prim_id, exponent in enumerate(pyscf_mol.bas_exp(bas_id)):
                    primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))


                basis_functions.append(BasisFunction(primitives, pyscf_mol.bas_ctr_coeff(bas_id),
                                                     center=pyscf_mol.bas_coord(bas_id),
                                                     label='{}:{}'.format(pyscf_mol.atom_symbol(pyscf_mol.bas_atom(bas_id)),
                                                                          bf_names[pyscf_mol.bas_angular(bas_id)]))
                                       )

        elif pyscf_mol.bas_angular(bas_id) == 2:  # d

            basis_list_temp = []
            for l_set in [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]:
                primitives = []
                for prim_id, exponent in enumerate(pyscf_mol.bas_exp(bas_id)):
                    primitives.append(PrimitiveGaussian(alpha=exponent, l=l_set))

                basis_list_temp.append(BasisFunction(primitives, pyscf_mol.bas_ctr_coeff(bas_id),
                                                     center=pyscf_mol.bas_coord(bas_id),
                                                     label='{}:{}'.format(pyscf_mol.atom_symbol(pyscf_mol.bas_atom(bas_id)),
                                                                          bf_names[pyscf_mol.bas_angular(bas_id)])))

            # d1 : xy
            d1 = basis_list_temp[3]
            basis_functions.append(d1)

            # d2 : yz
            d2 = basis_list_temp[5]
            basis_functions.append(d2)

            # d3 : 2*z2-x2-y2 (z^2?)
            d3 = 2 * basis_list_temp[2] - basis_list_temp[0] - basis_list_temp[1]
            norm = 1 / np.sqrt((d3 * d3).integrate)
            basis_functions.append(d3 * norm)

            # d4 : xz
            d4 = basis_list_temp[4]
            basis_functions.append(d4)

            # d5 : x2-y2
            d5 = basis_list_temp[0] - basis_list_temp[1]
            norm = 1/np.sqrt((d5*d5).integrate)
            basis_functions.append(d5 * norm)
        else:
            raise Exception('Not yet implemented for angular momentum > 2')

    return basis_functions


def get_basis_overlap_matrix(molecule_1, molecule_2, print_extra=False):
    """
    compute overlap matrix between the molecular orbitals of molecule_1 and molecule_1
    :param molecule_1: PyscfMolecularData object
    :param molecule_2: PyscfMolecularData object
    :param print_extra: set True for extra printing (for testig)
    :return: the overlap matrix (numpy array)
    """

    mol = molecule_1._pyscf_data['mol']
    mo_mol = molecule_1._pyscf_data['scf'].mo_coeff.T

    mol2 = molecule_2._pyscf_data['mol']
    mo_mol2 = molecule_2._pyscf_data['scf'].mo_coeff.T

    basis_functions = get_basis_set_pyscf(mol)

    orbitals = []
    for orbital_coeff in mo_mol:
        orbital = BasisFunction([], [])
        for bf, coeff in zip(basis_functions, orbital_coeff):
            orbital += bf * coeff

        orbitals.append(orbital)

    if print_extra:
        print('Basis functions')
        print(basis_functions)

        print('MO overlap')
        overlap_matrix = np.zeros((len(orbitals), len(orbitals)))
        for i, orb_i in enumerate(orbitals):
            for j, orb_j in enumerate(orbitals):
                overlap_matrix[i, j] = (orb_i * orb_j).integrate

        print(np.round(overlap_matrix, decimals=2))

    mol2.build()
    basis_functions_2 = get_basis_set_pyscf(mol2)

    orbitals_2 = []
    for orbital_coeff in mo_mol2:
        orbital = BasisFunction([], [])
        for bf, coeff in zip(basis_functions_2, orbital_coeff):
            orbital += bf * coeff
        orbitals_2.append(orbital)

    if print_extra:
        print('Basis functions 2')
        print(basis_functions)

        print('MO overlap 2')
        overlap_matrix = np.zeros((len(orbitals_2), len(orbitals_2)))
        for i, orb_i in enumerate(orbitals_2):
            for j, orb_j in enumerate(orbitals_2):
                overlap_matrix[i, j] = (orb_i * orb_j).integrate

        print(np.round(overlap_matrix, decimals=2))

    cross_overlap_matrix = np.zeros((len(orbitals), len(orbitals_2)))
    for i, orb_i in enumerate(orbitals):
        for j, orb_j in enumerate(orbitals_2):
            cross_overlap_matrix[i, j] = (orb_i * orb_j).integrate

    return cross_overlap_matrix


def get_operator_prefactors(operator):
    """
    separates operator from its coefficient and normalized form

    :param operator: the operator
    :return: coefficient, normalized operator
    """
    coeff = 0
    for t in operator.terms:
        coeff_t = operator.terms[t]
        coeff += np.conj(coeff_t) * coeff_t

    if operator.many_body_order() > 0:
        return np.sqrt(coeff), operator/np.sqrt(coeff)

    raise Exception('Cannot normalize 0 operator')


def antisymmetryze(total_ansatz):
    """
    antisymmetrize and clean (separate operator and coefficient)

    :param total_ansatz: fermion operators ansatz
    :return: list of coefficients, list of operators
    """

    total_ansatz = proper_order(total_ansatz)
    def is_antisymmetric(fermion):
        hermitian_fermion = -hermitian_conjugated(fermion)
        return normal_ordered(total_ansatz) == normal_ordered(hermitian_fermion)

    if not is_antisymmetric(total_ansatz):
        # anti-symmetrize
        total_ansatz = (total_ansatz - hermitian_conjugated(total_ansatz))

    # check antisymmetric
    hermitian_fermion = -hermitian_conjugated(total_ansatz)
    assert normal_ordered(total_ansatz) == normal_ordered(hermitian_fermion)

    list_op = []
    list_coeff = []
    list_check = []
    for term in total_ansatz:
        h_op = term - normal_ordered(hermitian_conjugated(term))
        if h_op not in list_check:
            list_check.append(h_op)
            coeff, op = get_operator_prefactors(h_op)
            list_op.append(op)
            list_coeff.append(coeff)

    return list_coeff, list_op


def cuttoff_op(list_coeff, list_op, max_op):
    indices = np.argsort(np.abs(list_coeff))[:max_op]
    indices = np.sort(indices)
    return [list_coeff[i] for i in indices], [list_op[i] for i in indices],


def prepare_ansatz_for_restart(operator_ansatz, max_val=1e-2, max_op=None, singlet_symmetry=False, ref_pool=None):
    """
    separate normalized part of operators from its coefficients.

    :param operator_ansatz: Fermion of Qubit operator
    :param max_val: maximum coefficient value used to remove negligible operators contributions
    :param max_op: maximum number of operators (ordered by coefficients norm)
    :return:
    """

    if ref_pool is not None:
        list_coeff, list_op = ansatz_projection_into_pool(ref_pool, operator_ansatz, max_val=max_val)
        if max_op is not None:
            list_coeff, list_op = cuttoff_op(list_coeff, list_op, max_op)
        return list_coeff, OperatorList(list_op, antisymmetrize=False)

    if isinstance(operator_ansatz, FermionOperator):
        reduced_ansatz = FermionOperator()
        for term in operator_ansatz.terms:
            coeff = operator_ansatz.terms[term]
            if abs(coeff) > max_val:
                reduced_ansatz += coeff * FermionOperator(term)
        list_coeff, list_op = antisymmetryze(reduced_ansatz)
        if max_op is not None:
            list_coeff, list_op = cuttoff_op(list_coeff, list_op, max_op)

        list_op = OperatorList(list_op, antisymmetrize=False)

    elif isinstance(operator_ansatz, QubitOperator):
        list_coeff = []
        list_op = []
        for term in operator_ansatz.terms:
            coeff = operator_ansatz.terms[term]
            norm = np.linalg.norm(coeff)
            if norm > max_val:
                list_coeff.append(norm)
                list_op.append(coeff/norm * QubitOperator(term))

        if max_op is not None:
            list_coeff, list_op = cuttoff_op(list_coeff, list_op, max_op)
        list_op = OperatorList(list_op)

    elif operator_ansatz == 0:
        list_op = OperatorList([])
        list_coeff = []

    else:
        raise Exception('Ansatz preparation for {} type not implemented'.format(type(operator_ansatz)))

    return list_coeff, list_op


def project_basis(ansatz, basis_overlap_matrix, n_orb_1=None, frozen_core_1=0, n_orb_2=None, frozen_core_2=0):
    """
    project ansatz expressed in one basis or molecular orbitals into another

    :param ansatz: ansatz in fermion operators
    :param basis_overlap_matrix: basis transformation overlap matrix (MObasis1 x MObasis2)
    :param n_orb_1: number of orbitals in origin basis
    :param frozen_core_1: frozen orbitals in origin basis
    :param n_orb_2: number of orbitals in target basis
    :param frozen_core_2: frozen orbitals in target basis
    :return: list of coefficients, list of operators
    """

    basis_overlap_matrix = basis_overlap_matrix[frozen_core_1:n_orb_1, frozen_core_2:n_orb_2]
    basis_overlap_matrix_spin = np.kron(basis_overlap_matrix, np.identity(2))

    if isinstance(ansatz, FermionOperator):
        Operator = FermionOperator
    elif isinstance(ansatz, QubitOperator):
        Operator = QubitOperator
    elif ansatz == 0:
        return 0
    else:
        raise Exception('Operator projection for {} type not implemented'.format(type(ansatz)))

    projected_ansatz = Operator()

    for term in ansatz.terms:
        op_coeff = ansatz.terms[term]
        total_fermion = []
        for iorb, itype in term:
            sum_fermion = Operator()

            for j_orb, coeff in enumerate(basis_overlap_matrix_spin[iorb]):
                sum_fermion += coeff * Operator((j_orb, itype))

            total_fermion.append(sum_fermion)

        if len(total_fermion) > 0:
            projected_ansatz += op_coeff * reduce(mul, total_fermion)

    return projected_ansatz


def operator_dot(operator_1, operator_2):
    """
    compute overlap between two operators

    :param operator_1: operator 1
    :param operator_2: operator 2
    :return:
    """
    sum = 0
    for term_op1, coeff_op1 in operator_1.terms.items():
        for term_op2, coeff_op2 in operator_2.terms.items():
            if term_op1 == term_op2:
                sum += coeff_op1 * coeff_op2
    return sum


def state_projection_into_pool(ansatz_ref, state_operators):
    """
    project state_operators (raw ansatz)  into a pool ansatz_ref (Operators list)

    :param ansatz_ref: reference ansatz (result will be expressed in this basis)
    :param state_operators: operators that define state (including coefficients)

    :return: coefficients, projected ansatz (operators list)
    """

    coefficients = []
    for op in ansatz_ref:
        overlap = operator_dot(op, state_operators)
        coefficients.append(overlap)

    return coefficients


if __name__ == '__main__':

    # molecule basis 1
    mol_1 = gto.Mole()
    mol_1.build(atom='''O  0 0 0; 
                      H  0 1 0; 
                      H  0 0 1
                      ''',
                basis='sto-3g')

    scf_1 = mol_1.RHF().run()

    # molecule basis 2
    mol_2 = mol_1.copy()
    mol_2.basis = '3-21g'
    mol_2.build()

    scf_2 = mol_2.RHF().run()

    from dataclasses import dataclass

    @dataclass
    class MoleculeData:
        _pyscf_data: dict

    molecule_1 = MoleculeData(_pyscf_data={'mol': mol_1, 'scf': scf_1})
    molecule_2 = MoleculeData(_pyscf_data={'mol': mol_2, 'scf': scf_2})

    basis_overlap_matrix = get_basis_overlap_matrix(molecule_1, molecule_2)

    print('MO cross overlap')
    print(np.round(basis_overlap_matrix, decimals=2))

    # test fermion ansatz
    from utils import get_uccsd_operators
    ansatz = get_uccsd_operators(n_electrons=2, n_orbitals=2)
    print('Original ansatz')
    print(ansatz)

    projected_ansatz = project_basis(ansatz, basis_overlap_matrix, n_orb_2=3)

    # print(projected_ansatz)

    print('final projected ansatz')
    list_coeff, list_op = prepare_ansatz_for_restart(projected_ansatz, max_val=1e-2)
    print(list_coeff)
    print(list_op)
