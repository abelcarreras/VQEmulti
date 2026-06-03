from vqemulti.optimizers import OptimizerParams
from vqemulti.ansatz import GenericAnsatz
from vqemulti.utils import log_message
import numpy as np
import scipy


def hi_vqe(hamiltonian,
           ansatz: GenericAnsatz,
           energy_simulator=None,
           energy_threshold=1e-4,
           optimizer_params=None,
           sqd_params=None,
           ):
    """
    Perform a Hi-VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: Ansatz object tooptimize
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param optimizer_params: Optimizer params object
    :param sqd_params: Sqd params dictionary
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

    if sqd_params is None:
        sqd_params = {}

    # store data at each evaluation
    parameters_iter = []
    def callback(xk):
        # print('called')
        parameters_iter.append(xk.copy().tolist())

    # Optimize the results from analytical calculation
    results = scipy.optimize.minimize(ansatz.get_sampled_energy,
                                      coefficients,
                                      (hamiltonian, energy_simulator, sqd_params),
                                      method=optimizer_params.method,
                                      options=optimizer_params.options,
                                      tol=energy_threshold,
                                      callback=callback,
                                      )

    ansatz.parameters = results.x

    # print('history: ', paramters_iter)
    return {'energy': results.fun,
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
    n_electrons = molecule.n_electrons + 1
    n_orbitals = molecule.n_orbitals
    multiplicity = molecule.multiplicity + 1


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
                    sqd_params=sqd_conf,
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

