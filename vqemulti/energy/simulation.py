from vqemulti.utils import fermion_to_qubit


def _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_eval_operator, simulator):
    """
    Obtain the hamiltonian expectation value for a given VQE state (reference + ansatz) and a hamiltonian

    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param qubit_eval_operator: operator to be evaluated in qubit operator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # circuit_depth = simulator.get_circuit_depth(ansatz_qubit, hf_reference_fock)
    # print('circuit_depth', circuit_depth)

    energy = simulator.get_state_evaluation(qubit_eval_operator, state_preparation_gates)

    return energy


def simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian expectation value for a given adaptVQE state (reference + ansatz) and a hamiltonian

    :param coefficients: adaptVQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit for adaptVQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    # evaluate hamiltonian
    energy = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian, simulator)

    return energy


def simulate_adapt_vqe_energy_square(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian square expectation value for a given adaptVQE state (reference + ansatz) and a hamiltonian

    :param coefficients: adaptVQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # define operator hamiltonian square
    qubit_hamiltonian_square = qubit_hamiltonian * qubit_hamiltonian

    # transform ansatz to qubit for adaptVQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    # evaluate hamiltonian square
    energy_square = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian_square, simulator)

    return energy_square


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian expectation value for a given VQE state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit for VQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=True)

    # evaluate hamiltonian
    energy = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian, simulator)

    return energy


def simulate_vqe_energy_square(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian square expectation value for a given VQE state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # define operator hamiltonian square
    qubit_hamiltonian_square = qubit_hamiltonian * qubit_hamiltonian

    # transform ansatz to qubit for VQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=True)

    # evaluate hamiltonian
    energy = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian_square, simulator)

    return energy


def simulate_vqe_variance(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian square expectation value for a given VQE state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit for VQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=True)

    # prepare gates
    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # energy = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    variance = simulator.get_state_evaluation_variance(qubit_hamiltonian, state_preparation_gates)

    return variance


def simulate_adapt_vqe_variance(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian square expectation value for a given VQE state (reference + ansatz) and a hamiltonian

    :param coefficients: VQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    # transform ansatz to qubit for VQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

    # prepare gates
    state_preparation_gates = simulator.get_preparation_gates(ansatz_qubit, hf_reference_fock)

    # energy = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    variance = simulator.get_state_evaluation_variance(qubit_hamiltonian, state_preparation_gates)

    return variance
