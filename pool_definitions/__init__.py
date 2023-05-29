from pool_definitions.singlet_sd import get_pool_singlet_sd
from pool_definitions.singlet_gsd import get_pool_singlet_gsd
from pool_definitions.spin_gsd import get_pool_spin_complement_gsd
from openfermion.transforms import jordan_wigner
from openfermion import QubitOperator, FermionOperator, normal_ordered
from openfermion.utils import hermitian_conjugated


# transform pool from fermion to operators (JW), normalize (to 1j), separate in terms and remove duplicates
def generate_jw_operator_pool(pool):

    qubit_pool = []
    for fermion_operator in pool:
        qubit_operator = jordan_wigner(fermion_operator)

        for pauli in qubit_operator.terms:
            qubit_operator = QubitOperator(pauli, 1j)
            if qubit_operator not in qubit_pool:
                qubit_pool.append(qubit_operator)

    return qubit_pool


class OperatorList:

    def __new__(cls, operators, **kwargs):
        if isinstance(operators, OperatorList):
            return operators

        return object.__new__(cls)

    def __init__(self, operators, normalize=False, antisymmetrize=True, split=False):
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

        if normalize:
            self._list = [op/c for op, c in zip(operators, self.operators_prefactors())]

        if antisymmetrize and self._type == FermionOperator:
            total_fermion = FermionOperator()
            for op in self._list:
                total_fermion += op

            def is_antisymmetric(fermion):
                hermitian_fermion = -hermitian_conjugated(fermion)
                return normal_ordered(total_fermion) == normal_ordered(hermitian_fermion)

            if not is_antisymmetric(total_fermion):
                # anti-symmetrize
                total_fermion = (total_fermion - hermitian_conjugated(total_fermion)) / 2

            # check antisymmetric
            hermitian_fermion = -hermitian_conjugated(total_fermion)
            assert normal_ordered(total_fermion) == normal_ordered(hermitian_fermion)

            self._list = []
            for term in total_fermion:
                h_op = term - hermitian_conjugated(term)
                if h_op not in self._list:
                    self._list.append(h_op)

        if split:
            split_list = []
            for op in self._list:
                for term in op.terms:
                    split_list.append(self._type(term))
            self._list = split_list


    def __str__(self):
        return self._list.__str__()

    def __getitem__(self, item):
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

    def get_quibits_list(self, transform='jw', normalize=False):
        """
        return qubits in the basis set
        :param transform:
        :return:
        """

        if self.get_operators_type() == QubitOperator:
            return self
            # return OperatorList([1j*QubitOperator(list(t.terms.keys())[0]) for t in self._list])

        if transform == 'jw':
            total_qubit = QubitOperator()
            for op in self._list:
                total_qubit += jordan_wigner(op)
        else:
            raise Exception('{} transform not available')

        if normalize:
            return OperatorList([QubitOperator(t) for t in total_qubit.terms])
        else:
            return OperatorList(total_qubit)

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

    def __mul__(self, other):

        if isinstance(other, (int, float, complex)):
            new_list = [op * other for op in self._list]

            return OperatorList(new_list)

        raise Exception('Not compatible operation')

    def __rmul__(self, other):
        return self.__mul__(other)

if __name__ == '__main__':

    from openfermion import QubitOperator, FermionOperator
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
    qubit_list = fermion_list.get_quibits_list(transform='jw')
    for op in qubit_list:
        sparse_operator = get_sparse_operator(op)
        exp_operator = scipy.sparse.linalg.expm(sparse_operator)
        exp_list_qubit.append(exp_operator)

