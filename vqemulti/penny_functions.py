import pennylane as qml
from pennylane import numpy as np
from openfermion import QubitOperator
from pennylane.ops.qubit.hamiltonian import OBS_MAP
from pennylane.operation import Observable, Tensor


def openfermion_to_penny(hamiltonian):

    def tens_prod(list_op):
        total = list_op[0]
        for op in list_op[1:]:
            total = total @ op
        return total

    def txt_to_pauli(str):

        num = str[0]
        op = str[1]
        if op == 'I':
            return qml.Identity(num)
        if op == 'X':
            return qml.PauliX(num)
        if op == 'Y':
            return qml.PauliY(num)
        if op == 'Z':
            return qml.PauliZ(num)

        raise Exception('Operator not recognized')

    def string_to_pauli(string):
        obs = []
        for str in string:
            obs.append(txt_to_pauli(str))
        if len(obs) == 0:
            obs.append(qml.Identity(0))
        obs = [tens_prod(obs)]

        return tens_prod(obs)

    coeff_list = []
    obs_list = []
    for term, coef in hamiltonian.terms.items():
        obs_list.append(string_to_pauli(term))
        coeff_list.append(coef)

    return qml.Hamiltonian(coeff_list, obs_list, grouping_type='qwc')


def penny_to_openfermion(hamiltonian):

    def wires_print(ob: Observable):
        """Function that formats the wires."""
        return ",".join(map(str, ob.wires.tolist()))

    of_qubit = 0
    for coeff, obs in zip(*hamiltonian.terms()):

        if isinstance(obs, Tensor):
            obs_strs = [f"{OBS_MAP.get(ob.name, ob.name)}{wires_print(ob)}" for ob in obs.obs]
            ob_str = " ".join(obs_strs)
        elif isinstance(obs, Observable):
            ob_str = f"{OBS_MAP.get(obs.name, obs.name)}{wires_print(obs)}"
        else:
            raise Exception('Transformation error')

        if ob_str == 'I0':
            of_qubit += float(coeff) * QubitOperator(' ')
        else:
            of_qubit += float(coeff) * QubitOperator(ob_str)

    return of_qubit


def test_tapering(hamiltonian, n_qubits, n_electrons):

    hamiltonian = openfermion_to_penny(hamiltonian)

    generators = qml.symmetry_generators(hamiltonian)
    paulixops = qml.paulix_ops(generators, n_qubits)

    for idx, generator in enumerate(generators):
        print(f"generator {idx+1}: {generator}, paulix_op: {paulixops[idx]}")

    paulix_sector = qml.qchem.optimal_sector(hamiltonian, generators, n_electrons)
    print(paulix_sector)

    H_tapered = qml.taper(hamiltonian, generators, paulixops, paulix_sector)
    print(H_tapered)


    H_sparse = qml.SparseHamiltonian(hamiltonian.sparse_matrix(), wires=hamiltonian.wires)
    H_tapered_sparse = qml.SparseHamiltonian(H_tapered.sparse_matrix(), wires=H_tapered.wires)

    print("Eigenvalues of H:\n", qml.eigvals(H_sparse, k=16))
    print("\nEigenvalues of H_tapered:\n", qml.eigvals(H_tapered_sparse, k=4))


    state_tapered = qml.qchem.taper_hf(generators, paulixops, paulix_sector,
                                       num_electrons=n_electrons, num_wires=len(hamiltonian.wires))
    print(state_tapered)
    print(H_tapered)

    return state_tapered, penny_to_openfermion(H_tapered)