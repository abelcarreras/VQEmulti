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
