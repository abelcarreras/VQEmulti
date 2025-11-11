from openfermion import get_sparse_operator
from openfermion import QubitOperator
import numpy as np
import scipy
import cirq



def sort_tuples(tuples):

    tuples = list(tuples)
    count = 1

    for index in range(len(tuples)):
        idx = np.argmax(np.array(tuples).T[0][index:])

        if index != idx + index:
            tuples[index], tuples[idx + index] = tuples[idx + index], tuples[index]
            count *= -1

    return tuple(tuples), count


# Example usage
tuples = ((3, 1), (4, 1), (0, 0), (1, 0))
s_tuples, count = sort_tuples(tuples)
print(s_tuples, count)

exit()


qubits = cirq.LineQubit.range(1)

# Create the gates for preparing the Hartree Fock ground state, that serves
# as a reference state the ansatz will act on
coefficient = np.pi*0.2

op = 1j * QubitOperator('X0 Y1')
operator_matrix = get_sparse_operator(coefficient * op)
# print(operator_matrix.toarray())

matrix = scipy.sparse.linalg.expm(operator_matrix)
print('exp:\n', np.round(matrix.toarray(), decimals=3))
print('-----')


rx = cirq.rx(-2*coefficient)(cirq.LineQubit(0))
ry = cirq.ry(-2*coefficient)(cirq.LineQubit(1))
identity = cirq.IdentityGate(1)(cirq.LineQubit(1))
circuit = cirq.Circuit(rx, ry)
print(circuit)
print(np.round(cirq.unitary(circuit), decimals=3))



from openfermion import get_sparse_operator
from openfermion import QubitOperator
import numpy as np
import scipy
import cirq


qubits = cirq.LineQubit.range(1)

# Create the gates for preparing the Hartree Fock ground state, that serves
# as a reference state the ansatz will act on


coefficient = 0.765  # test coeff

op = 1j * QubitOperator('X0')
operator_matrix = get_sparse_operator(coefficient * op)

matrix = scipy.sparse.linalg.expm(operator_matrix)
print('Matrix Operator:\n', np.round(matrix.toarray(), decimals=3))

rx = cirq.rx(-2*coefficient)(cirq.LineQubit(0))
circuit = cirq.Circuit(rx)
print('circuit')
print(circuit)

print('Rx Operator:\n', np.round(cirq.unitary(circuit), decimals=3))