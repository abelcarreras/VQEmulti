from vqemulti.utils import log_message
from collections import defaultdict
from vqemulti.utils import get_fock_space_vector, get_selected_ci_energy_dice, get_selected_ci_energy_qiskit
import numpy as np



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
        # if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            # print(fock_vector)
        configurations.append(fock_vector)

    # configurations = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=2, sample=0.01)[1]
    log_message('# configuration: {}'.format(len(configurations)), log_level=1)

    configurations = configuration_recovery(configurations, hamiltonian, n_electrons,
                                              multiplicity=0, n_max_diff=4, n_iter=8)

    log_message('# recovery conf: {}'.format(len(configurations)), log_level=1)

    log_message('start diagonalization ({})'.format(backend.lower()), log_level=1)
    if backend.lower() == 'dice':
        sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian)
    else:
        sqd_energy = get_selected_ci_energy_qiskit(configurations, hamiltonian)

    if return_samples:
        return sqd_energy, samples

    return sqd_energy


def generate_full_configurations(orbital_conf):

    full_configurations = set()
    for alpha in orbital_conf:
        for beta in orbital_conf:
            conf = []
            for a, b in zip(alpha, beta):
                conf += [a, b]
            full_configurations.add(tuple(conf))

    return list(full_configurations)


def joint_configurations(conf1, conf2):

    total_conf = set()
    for c in conf1:
        total_conf.add(tuple(c))
    for c in conf2:
        total_conf.add(tuple(c))
    return list(total_conf)


def configuration_recovery(configurations, hamiltonian, n_electrons, multiplicity=0, n_max_diff=4, n_iter=1):

    if multiplicity > 0:
        raise NotImplementedError('multiplicity must be 0!')

    n_electrons_alpha = n_electrons // 2
    n_electrons_beta = n_electrons // 2

    orbital_conf_good = set()
    orbital_conf_bad = set()

    for conf in configurations:
        alpha = tuple(conf[1::2])
        beta = tuple(conf[::2])

        if sum(alpha) == n_electrons_alpha:
            orbital_conf_good.add(alpha)
        else:
            orbital_conf_bad.add(alpha)

        if sum(beta) == n_electrons_beta:
            orbital_conf_good.add(beta)
        else:
            orbital_conf_bad.add(beta)

    full_configurations = generate_full_configurations(orbital_conf_good)
    log_message('# initial total unique conf: {}'.format(len(full_configurations)), log_level=1)

    for i_iter in range(n_iter):
        _, rdm = get_selected_ci_energy_dice(full_configurations, hamiltonian, return_density_matrix=True)
        prob_vec = np.diag(rdm)/n_electrons
        log_message('SCI orbital occupancy: {}'.format(prob_vec), log_level=1)

        # prob_vec = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001])
        # prob_vec = np.array([0.5, 0.1, 0.2, 0.0, 0.1, 0.2, 0.5, 0.0])
        n_orbitals = len(prob_vec)

        new_conf_list = []
        for conf in orbital_conf_bad:
            diff = n_electrons_alpha - sum(conf)
            new_conf = list(conf)
            for _ in range(n_max_diff):
                if diff > 0:
                    prob_vec_one = prob_vec / sum(prob_vec)
                    choice = np.random.choice(list(range(n_orbitals)), p=prob_vec_one)
                    if conf[choice] == 0:
                        new_conf[choice] = 1
                        if sum(new_conf) == n_electrons_alpha:
                            break
                else:
                    prob_vec_zero = (1-prob_vec) / sum(1-prob_vec)
                    choice = np.random.choice(list(range(n_orbitals)), p=prob_vec_zero)
                    if conf[choice] == 1:
                        new_conf[choice] = 0
                        if sum(new_conf) == n_electrons_alpha:
                            break

            if sum(new_conf) == n_electrons_alpha:
                new_conf_list.append(new_conf)

        # log_message('# iter {} recovery half conf: {}'.format(i_iter, len(new_conf_list)), log_level=1)

        recovered_full_conf = generate_full_configurations(new_conf_list)
        log_message('# iter {} recovery conf: {}'.format(i_iter, len(recovered_full_conf)), log_level=1)

        full_configurations = joint_configurations(full_configurations, recovered_full_conf)
        log_message('# iter {} total unique conf: {}'.format(i_iter, len(full_configurations)), log_level=1)

    return full_configurations

#from qiskit_addon_sqd.subsampling import postselect_and_subsample
#from qiskit_addon_sqd.configuration_recovery import recover_configurations


def get_variance(hamiltonia, ci_vector):
    pass

if __name__ == '__main__':

    with open('/Users/abel/PycharmProjects/VQEmulti/examples/configurations.dat', 'r') as f:
        raw_data = f.read().split('\n')

    configurations_bit = {}
    configurations = []
    for line in raw_data:#[:5]:
        conf, count = line.split()
        configurations_bit[conf] = int(count)
        configurations.append([int(c) for c in conf][::-1])


    #configuration_recovery(configurations, 10)

    a = configuration_recovery(configurations, None, 10)
    print(a)
    exit()