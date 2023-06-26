from vqemulti.utils import transform_to_scaled_qubit


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, qubit_hamiltonian, simulator):
    """
    Obtain the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state, using the CIRQ simulator.

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param qubit_hamiltonian: hamiltonian in qubits
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform ansatz to qubit (coefficients are included in qubits objects)
    ansatz_qubit = transform_to_scaled_qubit(ansatz, coefficients)

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # circuit_depth = simulator.get_circuit_depth(ansatz_qubit, hf_reference_fock)
    # print('circuit_depth', circuit_depth)

    energy = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    return energy
