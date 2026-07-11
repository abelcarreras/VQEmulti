from vqemulti.optimizers import OptimizerParams
from vqemulti.ansatz import GenericAnsatz
from vqemulti.utils import log_message
from vqemulti.sqd import simulate_energy_sqd, configuration_recovery, get_subspace_configurations, simple_filtering
from vqemulti.utils import get_selected_ci_energy_dice
import numpy as np
import scipy


def hi_vqe(hamiltonian,
           ansatz: GenericAnsatz,
           energy_simulator=None,
           energy_threshold=1e-4,
           optimizer_params=None,
           max_configurations=10000,
           n_partitions=1,
           weight_type=None,
           ):
    """
    Perform a Hi-VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: Ansatz object tooptimize
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param optimizer_params: Optimizer params object
    :param max_configurations: maximum number of configurations
    :param n_partitions: Number subspace partitions to evaluate
    :param weight_type: weight type (linear, geometric, exponential)
    :return: results dictionary
    """

    # set default optimizer params
    if optimizer_params is None:
        optimizer_params = OptimizerParams()

    log_message('optimizer params: ', optimizer_params, log_level=1)

    # initial guess
    coefficients = np.array(ansatz.parameters, dtype=float)

    # check if no coefficients
    if len(ansatz) == 0:
        energy = ansatz.get_energy(coefficients, hamiltonian, energy_simulator)
        return {'energy': energy, 'coefficients': [], 'ansatz': ansatz, 'f_evaluations': 0}

    # store data at each evaluation
    parameters_iter = []
    def callback(xk):
        # print('called')
        parameters_iter.append(xk.copy().tolist())

    if energy_simulator is None:
        raise Exception('sampled energy only works with simulator')

    n_alpha = sum(ansatz.reference_fock[::2])
    n_beta = sum(ansatz.reference_fock[1::2])

    n_electrons = n_alpha + n_beta
    multiplicity = (n_alpha - n_beta) + 1


    def get_sampled_energy(parameters, hamiltonian, sampling_simulator, return_std=False):
        """
        implementation of SQD energy as a function of ansatz parameters (for Hi-VQE like methods)

        :param parameters: ansatz paramters
        :param hamiltonian: hamiltonian in FermiOperator/InteractionOperator
        :param sampling_simulator: simulator for the sampling
        :param return_std:
        :return: SQD energy
        """

        # set paramters
        ansatz.parameters = parameters

        # get sampling
        samples = ansatz.get_sampling(sampling_simulator)

        # recovery
        # rec_samples = simple_filtering(samples, n_electrons, multiplicity=multiplicity)

        rec_samples = configuration_recovery(samples, hamiltonian, n_electrons,
                                             multiplicity=multiplicity,
                                             n_iter=4,
                                             n_max_diff=4,
                                             regularization_factor=0.7,
                                             max_configurations=max_configurations
                                             )

        # set subspace ranges
        n_conf_max = min(max_configurations, len(rec_samples))
        if n_partitions > 1:
            range_subspaces = np.linspace(1, n_conf_max, n_partitions, endpoint=True)
        else:
            range_subspaces = np.array([max_configurations], dtype=float)

        # define weighting function
        log_message('weight_type: {}'.format(weight_type), log_level=2)
        if weight_type is None:
            weights = np.ones_like(range_subspaces) # default

        elif weight_type == 'linear':
            weights = np.linspace(1.0, 0.0, n_partitions, endpoint=True)

        elif weight_type == 'geometric':
            param = 1e-3
            weights = np.geomspace(1.0, param, n_partitions)

        elif weight_type == 'exponential':
            param = 0.5
            x = np.linspace(0, 1, n_partitions)
            weights = np.exp(-param * x)
        else:
            raise Exception('Unknown weight type')

        # normalize weights
        weights /= weights.sum()

        # start SQD evaluations
        energies = []
        variances = []
        for n_conf in range_subspaces:
            log_message('n SQD configurations: ', int(n_conf), log_level=2)

            configurations = get_subspace_configurations(rec_samples, max_configurations=int(n_conf))

            if return_std:
                sqd_energy, extra_dice = get_selected_ci_energy_dice(configurations, hamiltonian, compute_variance=True)
                variances.append(extra_dice['variance'])
            else:
                sqd_energy = get_selected_ci_energy_dice(configurations, hamiltonian, compute_variance=False)

            energies.append(sqd_energy)

        energy = np.dot(energies, weights)

        if return_std:
            variance = np.dot(variances, weights)
            return energy, np.sqrt(variance)

        return energy


    # Optimize the results from analytical calculation
    results = scipy.optimize.minimize(get_sampled_energy,
                                      coefficients,
                                      (hamiltonian, energy_simulator),
                                      method=optimizer_params.method,
                                      options=optimizer_params.options,
                                      tol=energy_threshold,
                                      callback=callback,
                                      )

    ansatz.parameters = results.x

    # final converged energy
    energy = simulate_energy_sqd(ansatz, hamiltonian, energy_simulator, n_electrons,
                                 multiplicity=multiplicity,
                                 max_configurations=max_configurations,
                                 add_hf_configuration=True,
                                 recovery_type=1,
                                 return_extra=False)

    # print('history: ', parameters_iter)
    return {'energy': energy,
            'coefficients': results.x.tolist(),
            'ansatz': ansatz,
            'f_evaluations': results.nfev,
            'iterations': {'parameters': parameters_iter},
            }


if __name__ == '__main__':

    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.pool.singlet_sd import get_pool_singlet_sd
    from vqemulti.ansatz.unitary_jastrow import UnitaryCoupledJastrowAnsatz
    from vqemulti.preferences import Configuration
    import matplotlib.pyplot as plt

    Configuration().mapping = 'jw'
    Configuration().verbose = True
    Configuration().temp_dir = '/Users/abel/TEST/DICE'

    nitrogen = MolecularData(geometry=[('N', [0.0, 0.0, 0.0]),
                                       ('N', [2.0, 0.0, 0.0])],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='molecule')

    # run reference calculation
    molecule = run_pyscf(nitrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=4, n_orbitals=10, run_ccsd=True, run_casci=True)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = molecule.n_orbitals
    multiplicity = molecule.multiplicity


    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)
    print('multiplicity', multiplicity)


    hamiltonian = molecule.get_molecular_hamiltonian()
    # print(hamiltonian)

    print('n_qubits:', hamiltonian.n_qubits)


    print('\nJASTROW ansatz\n==============')

    ccsd = molecule._pyscf_data.get('ccsd', None)


    # Get UCCSD params
    uccsd_pool = get_pool_singlet_sd(n_electrons, n_orbitals)
    # uccsd_ansatz = []

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, multiplicity)
    print('hf reference', hf_reference_fock)

    # get ansatz
    from vqemulti.ansatz.exponential import ExponentialAnsatz

    initial_parameters = np.zeros_like(uccsd_pool)
    uccsd_ansatz = ExponentialAnsatz(initial_parameters, uccsd_pool, hf_reference_fock)


    ucja = UnitaryCoupledJastrowAnsatz(ccsd.t1, ccsd.t2, hf_reference_fock, n_terms=1, full_trotter=True)


    # Simulator
    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          shots=10000)

    #print(ucja.get_sampling(simulator))
    #exit()

    sqd_conf = {'multiplicity': multiplicity,
                'recovery_type': 1}


    # define optimizer
    opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

    print('Initialize VQE')
    result = hi_vqe(hamiltonian,
                    ucja,
                    energy_simulator=simulator,
                    optimizer_params=opt_cobyla,
                    )

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
    print('Energy CASCI: {:.8f}'.format(molecule.casci_energy))

    print('Num operators: ', len(result['ansatz']))
    #print('Ansatz:\n', result['ansatz']._operators)
    print('Coefficients:\n', result['coefficients'])
    repr(result)

    simulator.print_statistics()

    plt.plot()

