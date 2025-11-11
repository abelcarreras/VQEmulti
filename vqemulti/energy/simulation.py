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

    energy, std_error = simulator.get_state_evaluation(qubit_eval_operator, state_preparation_gates)

    return energy, std_error


def simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, return_std=False):
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
    energy, std_error = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian, simulator)

    if return_std:
        return energy, std_error

    return energy


def simulate_energy_sqd(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, n_electrons,
                        multiplicity=0,
                        generate_random=False,
                        backend='dice',
                        adapt=False,
                        return_samples=False):
    """
    Obtain the hamiltonian expectation value with SQD using a given adaptVQE state as reference.
    Only compatible with JW mapping!!

    :param coefficients: adaptVQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in InteractionOperator
    :param simulator: simulation object
    :param n_electrons: number of electrons
    :param multiplicity: multiplicity
    :param generate_random: generate random configuration distribution instead of simulation
    :param backend: backend to use for selected-CI  (dice, qiskit)
    :param adapt: True if adaptVQE False if VQE
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    from vqemulti.preferences import Configuration
    from vqemulti.utils import get_fock_space_vector, get_selected_ci_energy_dice, get_selected_ci_energy_qiskit
    from vqemulti.utils import get_dmrg_energy

    alpha_electrons = (multiplicity + n_electrons)//2
    beta_electrons = (n_electrons - multiplicity)//2

    # print('electrons: ', alpha_electrons, beta_electrons)

    from qiskit_addon_sqd.counts import generate_counts_uniform
    import numpy as np

    from openfermion.ops.representations import InteractionOperator
    if not isinstance(hamiltonian, InteractionOperator):
        # transform operator
        # from openfermion.transforms import get_interaction_operator
        # hamiltonian = get_interaction_operator(hamiltonian)
        raise Exception('Hamiltonian must be a InteractionOperator')

    # transform ansatz to qubit for VQE/adaptVQE
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=not adapt)

    if generate_random:
        samples = generate_counts_uniform(simulator._shots, len(hf_reference_fock))
    else:
        samples = simulator.get_sampling(ansatz_qubit, hf_reference_fock)

    if Configuration().verbose:
        print('samples ({}):'.format(len(samples)), samples)

    configurations = []
    for bitstring in samples.keys():
        fock_vector = get_fock_space_vector([1 if b == '1' else 0 for b in bitstring[::-1]])
        if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            # print(fock_vector)
            configurations.append(fock_vector)

    # configurations = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=2, sample=0.01)[1]

    if backend.lower() == 'dice':
        sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian)
    else:
        sqd_energy = get_selected_ci_energy_qiskit(configurations, hamiltonian)

    if return_samples:
        return sqd_energy, samples

    return sqd_energy

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


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, return_std=False):
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
    energy, std_error = _simulate_generic(ansatz_qubit, hf_reference_fock, qubit_hamiltonian, simulator)

    if return_std:
        return energy, std_error

    return energy


def simulate_vqe_energy_sqd(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator):
    """
    Obtain the hamiltonian expectation value with SQD using a given adaptVQE state as reference

    :param coefficients: adaptVQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in InteractionOperator
    :param simulator: simulation object
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs, solve_fermion
    from qiskit_addon_sqd.counts import generate_counts_uniform
    import numpy as np

    # transform ansatz to qubit for VQE (coefficients are included in qubits objects)
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=True)

    samples = simulator.get_sampling(ansatz_qubit, hf_reference_fock)
    # samples = generate_counts_uniform(simulator._shots, len(hf_reference_fock))
    print(samples)

    up_list = []
    down_list = []

    for bitstring in samples.keys():

        up_str = ''.join(
            '0' if i % 2 == 0 else bit
            for i, bit in enumerate(bitstring)
        )

        down_str = ''.join(
            bit if i % 2 == 0 else '0'
            for i, bit in enumerate(bitstring)
        )

        if up_str.count('1') == 2 and down_str.count('1') == 2:
            up_list.append(int(up_str, 2))
            down_list.append(int(down_str, 2))

    up = np.array(up_list, dtype=np.uint32)
    down = np.array(down_list, dtype=np.uint32)

    inv = np.argsort((0, 2, 3, 1))
    two_body_tensor_restored = hamiltonian.two_body_tensor.transpose(inv) * 2

    print('n_configurations: ', len(up))

    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion((up, down),
                                                           hamiltonian.one_body_tensor,
                                                           two_body_tensor_restored,
                                                           open_shell=True,
                                                           # spin_sq=0,
                                                           )

    return energy_sci + hamiltonian.constant


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
