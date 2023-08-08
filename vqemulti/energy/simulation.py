from vqemulti.utils import fermion_to_qubit


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the energy expectation value for a given state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # circuit_depth = simulator.get_circuit_depth(ansatz_qubit, hf_reference_fock)
    # print('circuit_depth', circuit_depth)

    energy = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    return energy

def simulate_vqe_energy_square(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the energy square expectation value for a given state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # circuit_depth = simulator.get_circuit_depth(ansatz_qubit, hf_reference_fock)
    # print('circuit_depth', circuit_depth)

    energy = simulator.get_state_evaluation(qubit_hamiltonian*qubit_hamiltonian, state_preparation_gates)

    return energy
