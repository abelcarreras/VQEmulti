from openfermion import normal_ordered, FermionOperator, QubitOperator
from openfermion.utils import hermitian_conjugated
from vqemulti.utils import normalize_operator, proper_order, fermion_to_qubit, get_string_from_fermionic_operator


class OperatorList:

    def __new__(cls, operators, **kwargs):
        if isinstance(operators, OperatorList):
            return operators

        return object.__new__(cls)

    def __init__(self, operators, normalize=False, antisymmetrize=True, spin_symmetry=False):
        """
        basis class to manage operators lists

        :param operators: list of Fermion/Qubit operators
        :param normalize: if True, divide operators in list elements and set coefficients to one
        :param antisymmetrize: if True modify fermion operators to be antisymmetric
        """
        if isinstance(operators, (list, tuple)):
            if len(operators) > 0:
                self._type = type(operators[0])
            else:
                # empty list
                self._type = None
            self._list = operators
        else:
            self._type = type(operators)
            self._list = [op for op in operators]

        if antisymmetrize and self._type == FermionOperator:
            total_fermion = FermionOperator()
            self._original_list = self._list
            self._list = []
            for op in self._original_list:

                def is_anti_hermitian(fermion):
                    hermitian_fermion = -hermitian_conjugated(fermion)
                    return normal_ordered(fermion) == normal_ordered(hermitian_fermion)

                if is_anti_hermitian(op):
                    self._list.append(op)

                else:
                    # anti-symmetrize
                    total_fermion = (total_fermion - hermitian_conjugated(total_fermion))

                    # check antisymmetric
                    anti_hermitian_fermion = -hermitian_conjugated(total_fermion)
                    assert normal_ordered(total_fermion) == normal_ordered(anti_hermitian_fermion)

                    self._list.append(anti_hermitian_fermion)

        if normalize:
            self._list = [normalize_operator(op) for op in self._list]
            # self._list = [op/c for op, c in zip(operators, self.operators_prefactors())]


    def __str__(self):
        return self._list.__str__()

    def __getitem__(self, item):
        if isinstance(self._list[item], list):
            return OperatorList(self._list[item], antisymmetrize=False, normalize=False)
        else:
            return self._list[item]
    def __len__(self):
        return len(self._list)

    def append(self, operator, join=False):
        if self._type is None:
            self._type = type(operator)
        else:
            if self._type != type(operator):
                raise Exception('Operator not compatible with this list')

        if join:
            self._list += [op for op in operator]
        else:
            self._list.append(operator)

    def get_operators_type(self):
        return self._type

    def get_quibits_list(self, normalize=False, reorganize=True):
        """
        return qubits in the basis set
        :param mapping:
        :return:
        """

        if self.get_operators_type() == QubitOperator:
            return self
            # return OperatorList([1j*QubitOperator(list(t.terms.keys())[0]) for t in self._list])

        total_qubit = QubitOperator()
        if reorganize:
            for op in self._list:
                total_qubit += fermion_to_qubit(op)
            return OperatorList(total_qubit, normalize=normalize)
        else:
            total_qubit = []
            for op in self._list:
                total_qubit.append(fermion_to_qubit(op))
            return OperatorList(total_qubit, normalize=normalize)

    def get_expanded_list(self):
        """
        return qubits in the basis set
        :return:
        """

        expanded_list = []
        for element in self._list:
            for t in element.terms:
                expanded_list.append(self._type(t))

        return OperatorList(expanded_list)

    def operators_prefactors(self):

        prefactors = []
        for op in self._list:
            if self.get_operators_type() == QubitOperator:
                prefactors.append((list(op.terms.values())[0]).real)
            else:
                prefactors.append((list(op.terms.values())[0]).real)

        return prefactors

    def scale_vector(self, coefficients):
        if len(coefficients) != len(self._list):
            raise Exception('Size do not match')

        new_list = []
        for c, op in zip(self._list, coefficients):
            new_list.append(c * op)
        self._list = new_list

    def copy(self):
        from copy import deepcopy
        return OperatorList(deepcopy(self._list), antisymmetrize=False)

    def transform_to_scaled_qubit(self, coefficients, join=False):

        ansatz = self.copy()
        ansatz.scale_vector(coefficients)
        ansatz_qubit = ansatz.get_quibits_list(reorganize=join)

        if join:
            sum_ansatz = sum(ansatz_qubit)
            if sum_ansatz != 0:
                return OperatorList([sum_ansatz])

        return ansatz_qubit

    def print_compact_representation(self):
        for op in self._list:
            print(get_string_from_fermionic_operator(op))

    def get_index(self, operators_pool):
        indeces = []
        for operator in self._list:
            for j in range(len(operators_pool)):
                if operator == operators_pool[j]:
                    indeces.append(j)
        return indeces




    def __mul__(self, other):

        if isinstance(other, (int, float, complex)):
            new_list = [op * other for op in self._list]

            return OperatorList(new_list)

        raise Exception('Not compatible operation')

    def __rmul__(self, other):
        return self.__mul__(other)

if __name__ == '__main__':

    from openfermion import get_sparse_operator
    import scipy

    fermion_op = 2*FermionOperator(((3, 1), (1, 0))) + 3*FermionOperator(((2, 0), (1, 1)))

    exp_list_fermion = []
    fermion_list = OperatorList(fermion_op, antisymmetrize=True)
    for op in fermion_list:
        sparse_operator = get_sparse_operator(op)
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)
        exp_list_fermion.append(exp_operator)

    exp_list_qubit = []
    qubit_list = fermion_list.get_quibits_list()
    for op in qubit_list:
        sparse_operator = get_sparse_operator(op)
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)
        exp_list_qubit.append(exp_operator)

