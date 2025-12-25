from vqemulti.utils import log_message


def simulate_energy_sqd(ansatz, hamiltonian, simulator, n_electrons,
                        multiplicity=0,
                        generate_random=False,
                        backend='dice',
                        return_samples=False):
    """
    Obtain the hamiltonian expectation value with SQD using a given adaptVQE state as reference.
    Only compatible with JW mapping!!

    :param ansatz: ansatz object
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
    #ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=not adapt)

    log_message('start sampling ({})'.format(simulator), log_level=1)
    if generate_random:
        samples = generate_counts_uniform(simulator._shots, ansatz.n_qubits)
    else:
        samples = ansatz.get_sampling(simulator)

    log_message('# samples: {}'.format(len(samples)), log_level=1)

    configurations = []
    for bitstring in samples.keys():
        fock_vector = get_fock_space_vector([1 if b == '1' else 0 for b in bitstring[::-1]])
        if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            # print(fock_vector)
            configurations.append(fock_vector)

    # configurations = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=2, sample=0.01)[1]
    log_message('# configuration: {}'.format(len(configurations)), log_level=1)

    log_message('start diagonalization ({})'.format(backend.lower()), log_level=1)
    if backend.lower() == 'dice':
        sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian)
    else:
        sqd_energy = get_selected_ci_energy_qiskit(configurations, hamiltonian)

    if return_samples:
        return sqd_energy, samples

    return sqd_energy
