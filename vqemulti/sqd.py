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

    configurations = configuration_recovery_2(configurations, hamiltonian, n_electrons,
                                              multiplicity=0, n_max_diff=4, n_iter=8)

    log_message('# recovery conf: {}'.format(len(configurations)), log_level=1)

    log_message('start diagonalization ({})'.format(backend.lower()), log_level=1)
    if backend.lower() == 'dice':
        sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian, stream_output=True)
    else:
        sqd_energy = get_selected_ci_energy_qiskit(configurations, hamiltonian)

    if return_samples:
        return sqd_energy, samples

    return sqd_energy


def configuration_recovery(configurations, n_electrons, multiplicity=0):
    n_electrons_alpha = n_electrons //2
    n_electrons_beta = n_electrons //2

    def hamming_distance(det1, det2):
        return (det1 ^ det2).bit_count()

    def get_bit(bitstring, index):
        return (bitstring >> index) & 1

    def flip_bit(bitstring, index):
        return bitstring ^ (1 << index)

    def add_one(bitstring, index):
        return bitstring | (1 << index)

    def add_zero(bitstring, index):
        return bitstring & ~(1 << index)

    def hamming_weight(bitstring):
        return bitstring.bit_count()

    def det_to_string(det, n_orbitals):
        return format(det, f"0{n_orbitals}b")

    def orbital_probabilities(orbital_conf, n_orbitals):

        probs = np.zeros(n_orbitals, dtype=float)

        norm = 0
        for conf, v in orbital_conf.items():
            # print(det_to_string(conf, n_orbitals))
            for i in range(n_orbitals):
                probs[i] += ((conf >> i) & 1 ) * v
            norm += v

        probs /= norm
        return probs#[::-1]


    n_orbitals = len(list(configurations.keys())[0])//2
    good_conf = defaultdict(int)
    bad_conf = defaultdict(int)
    for k, v in configurations.items():
        alpha = int(k[1::2], 2)
        beta = int(k[::2], 2)

        if hamming_weight(alpha) == n_electrons_alpha:
            good_conf[alpha] += v
        else:
            bad_conf[alpha] += v


        if hamming_weight(beta) == n_electrons_beta:
            good_conf[beta] += v
        else:
            bad_conf[beta] += v

    print(good_conf)
    print('good_conf :', len(good_conf))

    for _ in range(2):
        prob_vec = orbital_probabilities(good_conf, n_orbitals)

        def fix_conf(vec, prob_vec, h_distance, tolerance=5e-3):
            indices = np.argsort(prob_vec)
            ref_prob = None
            new_vec_list = []
            if h_distance > 0:
                for i in indices[::-1]:
                    #print('here', ref_prob, abs(ref_prob - prob_vec[i]))
                    # print(i, 'val:', get_bit(vec, i))
                    if not get_bit(vec, i):

                        if ref_prob is not None and abs(ref_prob - prob_vec[i]) > tolerance:
                            break

                        # print('add 1 in ', i)
                        new_vec_list.append(add_one(vec, i))

                        ref_prob = prob_vec[i]

            else:
                ref_prob = prob_vec[indices[0]]
                for i in indices:
                    if get_bit(vec, i):
                        if ref_prob is not None and abs(ref_prob - prob_vec[i]) > tolerance:
                            break

                        new_vec_list.append(add_zero(vec, i))
                        ref_prob = prob_vec[i]

            return new_vec_list

        for k, v in bad_conf.items():
            h_distance = n_electrons_alpha - hamming_weight(k)
            if abs(h_distance) < 2:
                #print('in->', det_to_string(k,n_orbitals), h_distance)
                new_conf_list = fix_conf(k, prob_vec, h_distance)

                for conf in new_conf_list:
                    #print('out->', det_to_string(conf, n_orbitals), h_distance)
                    good_conf[conf] += v

                    assert hamming_weight(conf) == n_electrons_alpha

        print(good_conf)
        print('good_conf_i :', len(good_conf))

    sorted_configurations = sorted(good_conf.items(),
                                   key=lambda item: item[1],
                                   reverse=True)

    for k, v in sorted_configurations:
        print('{} {}'.format(det_to_string(k, n_orbitals), v))

    exit()


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


def configuration_recovery_2(configurations, hamiltonian, n_electrons, multiplicity=0, n_max_diff=4, n_iter=1):
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

from qiskit_addon_sqd.subsampling import postselect_and_subsample
from qiskit_addon_sqd.configuration_recovery import recover_configurations


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

    a = configuration_recovery_2(configurations, None, 10)
    print(a)
    exit()