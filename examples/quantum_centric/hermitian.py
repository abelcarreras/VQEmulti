from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

op = FermionOperator('0^ 0 2^ 2')

print('operator: ', op)
# print('->', op + hermitian_conjugated(op))

print('hermitian?', (op - hermitian_conjugated(op)).isclose(FermionOperator.zero(), 1e-4))
print('hermitian (normal order)?', normal_ordered(op - hermitian_conjugated(op)).isclose(FermionOperator.zero(), 1e-4))

op = 1j*op
print('anti-hermitian?', normal_ordered(op + hermitian_conjugated(op)).isclose(FermionOperator.zero(), 1e-4))
