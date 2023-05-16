from pool_definitions.singlet_sd import get_pool_singlet_sd
from pool_definitions.singlet_gsd import get_pool_singlet_gsd
from pool_definitions.spin_gsd import get_pool_spin_complement_gsd
from openfermion.transforms import jordan_wigner
from openfermion import QubitOperator

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

    def __new__(cls, operators):
        if isinstance(operators, OperatorList):
            return operators

        return object.__new__(cls)

    def __init__(self, operators):
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

    def __str__(self):
        return self._list.__str__()

    def __getitem__(self, item):
        return self._list[item]

    def __len__(self):
        return len(self._list)

    def append(self, operator):
        if self._type is None:
            self._type = type(operator)
        else:
            if self._type != type(operator):
                raise Exception('Operator not compatible with this list')

        self._list += [op for op in operator]

    def get_operators_type(self):
        return self._type

    def get_quibits_list(self, transform='jw'):
        """
        return qubits in the basis set
        :param transform:
        :return:
        """

        if self.get_operators_type() == QubitOperator:
            return self
            # return OperatorList([1j*QubitOperator(list(t.terms.keys())[0]) for t in self._list])

        if transform == 'jw':
            from openfermion.transforms import jordan_wigner
            total_qubit = QubitOperator()
            for op in self._list:
                print(op)
                total_qubit += jordan_wigner(op)
        else:
            raise Exception('{} transform not available')

        return OperatorList([1j*QubitOperator(t) for t in total_qubit.terms])

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
                prefactors.append((list(op.terms.values())[0]/1j).real)
            else:
                prefactors.append((list(op.terms.values())[0]).real)

        return prefactors