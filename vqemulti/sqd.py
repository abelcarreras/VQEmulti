from vqemulti.utils import log_message, log_section
from collections import defaultdict
from vqemulti.utils import get_fock_space_vector, get_selected_ci_energy_dice, get_selected_ci_energy_qiskit
import numpy as np



def simulate_energy_sqd(ansatz, hamiltonian, simulator, n_electrons,
                        multiplicity=0,
                        max_configurations=None,
                        add_hf_configuration=False,
                        generate_random=False,
                        backend='dice',
                        return_extra=False,
                        compute_variance=False,
                        recovery_type=2,
                        hf_bias=0.0):
    """
    Obtain the hamiltonian expectation value with SQD using a given adaptVQE state as reference.
    Only compatible with JW mapping!!

    :param ansatz: ansatz object
    :param hamiltonian: hamiltonian in InteractionOperator
    :param simulator: simulation object
    :param n_electrons: number of electrons
    :param multiplicity: multiplicity
    :param max_configurations: maximum number of configurations to diagonalize
    :param add_hf_configuration: add HF configuration to Selected-CI
    :param generate_random: generate random configuration distribution instead of simulation
    :param backend: backend to use for selected-CI  (dice, qiskit)
    :param compute_variance: compute projected variance of the state
    :param return_extra: return extra computed information
    :param recovery_type: select recovery function (1: test recovery 2: qiskit-like recovery 0: No recovery)
    :param hf_bias: favor near-HF configurations (must be positive!)
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    from vqemulti.utils import get_dmrg_energy

    alpha_electrons = (multiplicity + n_electrons)//2
    beta_electrons = (n_electrons - multiplicity)//2

    # print('electrons: ', alpha_electrons, beta_electrons)


    log_message('start sampling ({})'.format(simulator), log_level=1)
    if generate_random:
        from qiskit_addon_sqd.counts import generate_counts_uniform
        samples = generate_counts_uniform(simulator._shots, ansatz.n_qubits)
    else:
        samples = ansatz.get_sampling(simulator)

    log_message('# samples: {}'.format(len(samples)), log_level=1)

    if log_section(log_level=1):
        print('\noriginal sample list')
        sorted_configurations = sorted(samples.items(),
                                       key=lambda item: item[1],
                                       reverse=True)
        for k, v in sorted_configurations:
            print('{} {}'.format(k, v))


    # configurations = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=2, sample=0.01)[1]
    log_message('# configurations: {}'.format(len(samples)), log_level=1)

    if recovery_type == 1:
        rec_samples = configuration_recovery(samples, hamiltonian, n_electrons,
                                             multiplicity=0, n_max_diff=4, n_iter=4,
                                             max_configurations=max_configurations,
                                             hf_bias=hf_bias)

    elif recovery_type == 2:
        rec_samples = configuration_recovery_2(samples, hamiltonian, n_electrons,
                                             multiplicity=0, n_iter=4,
                                             max_configurations=max_configurations)

    else:
        rec_samples = simple_filtering(samples, n_electrons, multiplicity=0)


    log_message('# recovery conf: {}'.format(len(rec_samples)), log_level=1)
    log_message('start diagonalization ({})'.format(backend.lower()), log_level=1)


    if log_section(log_level=1):
        print('\nrecovered sample list')
        sorted_configurations = sorted(rec_samples.items(),
                                       key=lambda item: item[1],
                                       reverse=True)
        for k, v in sorted_configurations:
            print('{} {}'.format(k, v))

    configurations = get_subspace_configurations(rec_samples, max_configurations,
                                                 add_hf_configuration=add_hf_configuration)

    extra = {'variance': None, 'configurations': configurations, 'rec_samples': rec_samples}
    if backend.lower() == 'dice':
        if compute_variance:
            sqd_energy, extra_dice = get_selected_ci_energy_dice(configurations, hamiltonian, compute_variance=True)
            extra.update(extra_dice)
        else:
            sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian, compute_variance=False)
    else:
        sqd_energy = get_selected_ci_energy_qiskit(configurations, hamiltonian)

    if return_extra:
        return sqd_energy, extra

    return sqd_energy


def simple_filtering(samples, n_electrons, multiplicity=0):

    if multiplicity > 0:
        raise NotImplementedError('multiplicity must be 0!')

    n_electrons_alpha = n_electrons // 2
    n_electrons_beta = n_electrons // 2

    orbital_conf_good = defaultdict(int)

    for bistring, count in samples.items():

        alpha = bistring[1::2]
        beta = bistring[::2]

        if alpha.count("1") == n_electrons_alpha:
            orbital_conf_good[alpha] += count / 2

        if beta.count("1") == n_electrons_beta:
            orbital_conf_good[beta] += count / 2

    return generate_full_samples(orbital_conf_good, orbital_conf_good)

def get_subspace_configurations(samples, max_configurations, add_hf_configuration=False):

    # order by frequency
    sorted_samples = sorted(samples.items(), key=lambda item: item[1], reverse=True)

    configurations = []
    for bitstring, c in sorted_samples:
        fock_vector = get_fock_space_vector([1 if b == '1' else 0 for b in bitstring[::-1]])
        configurations.append(fock_vector)

    # truncate the lowest frequency configurations
    if max_configurations is not None and len(configurations) > max_configurations:
        log_message('Max conf. exceeded. Sampling conf.: {}'.format(max_configurations), log_level=1)
        configurations = configurations[:max_configurations]

    if add_hf_configuration:
        log_message('Adding HF configuration', log_level=1)
        n_qubits = len(configurations[0])
        n_electrons = sum(configurations[0])
        hf_conf = [1] * n_electrons + [0] * (n_qubits - n_electrons)
        if hf_conf not in configurations:
            configurations.append(hf_conf)

    return configurations


def generate_full_samples(orbital_sample_alpha, orbital_sample_beta):

    full_samples = defaultdict(int)
    for alpha, ca in orbital_sample_alpha.items():
        for beta, cb in orbital_sample_beta.items():
            conf = ''.join(a + b for a, b in zip(alpha, beta))
            full_samples[conf] += ca * cb

    return full_samples


def scale_probability(prob_vector, n_occupied, strength=1.0):
    """
    scale probability density by a sigmoid function to favor lower energy configurations

    :param prob_vector: probability vector
    :param n_occupied: number of occupied orbitals
    :param strength: parameter to adjust the strength of HF-close configurations
    :return: new probability vector
    """
    def sigmoid(x, position=0, k=1.0):
        return 1.0 / (1.0 + np.exp(-k*(x-position)))

    def exponential(x, position=0.0, k=1.0):
        return np.exp(k * (x - position))

    n_orb = len(prob_vector)
    prob_vector_b = [float(sigmoid(i+0.5, position=n_occupied, k=strength)) for i in range(n_orb)]

    prob_vector = np.multiply(prob_vector, prob_vector_b)
    # prob_vector = np.convolve(prob_vector, prob_vector_b, mode='same')
    prob_vector = prob_vector / np.sum(prob_vector)

    return prob_vector

def get_prob_diff(conf, prob_vec_bistring):
    return np.abs(np.subtract(prob_vec_bistring, [int(i) for i in conf]))

def get_indices(bitstring, index):
        return [i for i, bit in enumerate(bitstring) if bit == str(index)]

def configuration_recovery(samples, hamiltonian, n_electrons, multiplicity=0, n_max_diff=4, n_iter=1, max_configurations=None, hf_bias=0.0):


    def get_electrons(bitsring):
        return len

    if multiplicity > 0:
        raise NotImplementedError('multiplicity must be 0!')

    n_electrons_alpha = n_electrons // 2
    n_electrons_beta = n_electrons // 2

    orbital_conf_good = defaultdict(int)
    orbital_conf_bad = defaultdict(int)

    for bistring, count in samples.items():

        alpha = bistring[1::2]
        beta = bistring[::2]

        if alpha.count("1") == n_electrons_alpha:
            orbital_conf_good[alpha] += count / 2
        else:
            orbital_conf_bad[alpha] += count / 2

        if beta.count("1") == n_electrons_beta:
            orbital_conf_good[beta] += count / 2
        else:
            orbital_conf_bad[beta] += count / 2

    full_samples_original = generate_full_samples(orbital_conf_good, orbital_conf_good)
    log_message('# original total unique conf: {}'.format(len(full_samples_original)), log_level=1)

    def set_bit(bitstring, position, bit):
        return bitstring[:position] + bit + bitstring[position + 1:]

    full_samples = full_samples_original.copy()
    for i_iter in range(n_iter):
        sampled_configurations = get_subspace_configurations(full_samples, max_configurations,
                                                             add_hf_configuration=True)

        results = get_selected_ci_energy_dice(sampled_configurations, hamiltonian, compute_density_matrix=True)[1]
        prob_vec = np.diag(results['1rdm'])/n_electrons
        log_message('SCI orbital occupancy: {}'.format(prob_vec), log_level=1)

        prob_vec_bistring = prob_vec[::-1] # set in bistring order
        prob_vec_bit_one = scale_probability(prob_vec_bistring, n_electrons//2, strength=hf_bias)
        prob_vec_bit_zero = scale_probability(1-prob_vec_bistring, n_electrons//2, strength=-hf_bias)
        print('prob_vec_one', prob_vec_bit_one)
        print('prob_vec_zero', prob_vec_bit_zero)

        n_orbitals = len(prob_vec_bistring)

        # new_conf_list = []
        new_conf_dict = defaultdict(int)
        print('prob_vec_bistring', prob_vec_bistring)

        for conf, c in orbital_conf_bad.items():
            diff = n_electrons_alpha - conf.count("1")
            new_conf = str(conf)
            for _ in range(n_max_diff):
                if diff > 0:
                    #prob_vec_one = prob_vec_bistring / sum(prob_vec_bistring)
                    #prob_vec_one = (1-prob_vec_bistring) / sum(1-prob_vec_bistring)
                    # print('prob_vec_one', prob_vec_one)
                    choice = np.random.choice(list(range(n_orbitals)), p=prob_vec_bit_one)
                    if conf[choice] == "0":
                        #print('1_ini', new_conf)
                        new_conf = set_bit(new_conf, choice, "1")
                        #print('1_fin', new_conf)
                        if new_conf.count("1") == n_electrons_alpha:
                            break
                else:
                    #prob_vec_zero = (1-prob_vec_bistring) / sum(1-prob_vec_bistring)
                    #prob_vec_zero = prob_vec_bistring / sum(prob_vec_bistring)
                    # print('prob_vec_zero', prob_vec_zero)
                    choice = np.random.choice(list(range(n_orbitals)), p=prob_vec_bit_zero)
                    if conf[choice] == "1":
                        #print('2_ini', new_conf)
                        new_conf = set_bit(new_conf, choice, "0")
                        #print('2_fin', new_conf)
                        if new_conf.count("1") == n_electrons_alpha:
                            break

            if new_conf.count("1") == n_electrons_alpha:
                new_conf_dict[new_conf] += c

        # log_message('# iter {} recovery half conf: {}'.format(i_iter, len(new_conf_list)), log_level=1)

        recovered_full_samples = generate_full_samples(new_conf_dict, new_conf_dict)

        log_message('# iter {} recovery conf: {}'.format(i_iter, len(recovered_full_samples)), log_level=1)

        # add recovered samples to full_samples
        full_samples = full_samples_original.copy()
        for key, value in recovered_full_samples.items():
            full_samples[key] += value

        log_message('# iter {} total unique conf: {}'.format(i_iter, len(full_samples)), log_level=1)

    return full_samples


def configuration_recovery_2(samples,
                             hamiltonian,
                             n_electrons,
                             multiplicity=0,
                             n_iter=1,
                             max_configurations=None,
                             regularization_factor=0.7):

    def regularization(prob_vect, factor=0.2):
        return np.array([x if x > factor else 0 for x in prob_vect])

    if multiplicity > 0:
        raise NotImplementedError('multiplicity must be 0!')

    n_electrons_alpha = n_electrons // 2
    n_electrons_beta = n_electrons // 2

    orbital_conf_good = defaultdict(int)
    orbital_conf_bad = defaultdict(int)

    for bistring, count in samples.items():

        alpha = bistring[1::2]
        beta = bistring[::2]

        if alpha.count("1") == n_electrons_alpha:
            orbital_conf_good[alpha] += count / 2
        else:
            orbital_conf_bad[alpha] += count / 2

        if beta.count("1") == n_electrons_beta:
            orbital_conf_good[beta] += count / 2
        else:
            orbital_conf_bad[beta] += count / 2

    full_samples_original = generate_full_samples(orbital_conf_good, orbital_conf_good)
    log_message('# original total unique conf: {}'.format(len(full_samples_original)), log_level=1)

    def set_bit(bitstring, position, bit):
        return bitstring[:position] + bit + bitstring[position + 1:]

    full_samples = full_samples_original.copy()
    for i_iter in range(n_iter):
        sampled_configurations = get_subspace_configurations(full_samples, max_configurations,
                                                             add_hf_configuration=True)

        results = get_selected_ci_energy_dice(sampled_configurations, hamiltonian, compute_density_matrix=True)[1]
        prob_vec = np.diag(results['1rdm'])/2
        # prob_vec = [1, 1, 1, 0, 0, 0]
        log_message('SCI orbital occupancy: {}'.format(prob_vec), log_level=1)

        prob_vec_bistring = prob_vec[::-1] # set in bistring order

        new_conf_dict = defaultdict(int)

        for conf, c in orbital_conf_bad.items():
            diff = n_electrons_alpha - conf.count("1")
            prob_diff = get_prob_diff(conf, prob_vec_bistring)
            prob_diff = regularization(prob_diff, factor=regularization_factor)

            new_conf = str(conf)
            if diff > 0:
                indices_occupied = get_indices(new_conf, 0)
                p_choice = prob_diff[indices_occupied]
                if np.sum(p_choice) == 0:
                    continue

                p_choice = p_choice/np.sum(p_choice)

                try:
                    indices_to_flip = np.random.choice(indices_occupied, size=abs(diff), replace=False, p=p_choice)
                except ValueError:
                    continue

                for choice in indices_to_flip:
                    new_conf = set_bit(new_conf, choice, "1")

            else:
                indices_occupied = get_indices(new_conf, 1)
                p_choice = prob_diff[indices_occupied]

                if np.sum(p_choice) == 0:
                    continue

                p_choice = p_choice/np.sum(p_choice)
                try:
                    indices_to_flip = np.random.choice(indices_occupied, size=abs(diff), replace=False, p=p_choice)
                except ValueError:
                    continue

                for choice in indices_to_flip:
                    new_conf = set_bit(new_conf, choice, "0")

            if new_conf.count("1") == n_electrons_alpha:
                new_conf_dict[new_conf] += c

        # log_message('# iter {} recovery half conf: {}'.format(i_iter, len(new_conf_list)), log_level=1)

        recovered_full_samples = generate_full_samples(new_conf_dict, new_conf_dict)

        log_message('# iter {} recovery conf: {}'.format(i_iter, len(recovered_full_samples)), log_level=1)

        # add recovered samples to full_samples
        # full_samples = defaultdict(int)
        full_samples = full_samples_original.copy()
        for key, value in recovered_full_samples.items():
            full_samples[key] += value


        log_message('# iter {} total unique conf: {}'.format(i_iter, len(full_samples)), log_level=1)

    return full_samples

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
