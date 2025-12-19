from vqemulti.utils import get_string_from_fermionic_operator
from vqemulti.pool.tools import OperatorList
from vqemulti.errors import NotConvergedError, Converged
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.method.adapt_vanila import AdapVanilla
from vqemulti.vqe import vqe


def adaptVQE(hamiltonian,
             operators_pool,
             adapt_ansatz,
             energy_threshold=0.0001,
             max_iterations=50,
             method=None,
             energy_simulator=None,
             reference_dm=None,
             optimizer_params=OptimizerParams(),
             ):
    """
    Perform an adaptVQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param operators_pool: fermionic operators pool
    :param adapt_ansatz: ExponentialProduct ansatz
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param max_iterations: max number of adaptVQE iterations
    :param method: Method used to update the ansatz
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param gradient_simulator: Simulator object used to obtain the gradient, if None do not use simulator (exact)
    :param reference_dm: reference density matrix (ideally from fullCI) that is used to compute the quantum fidelity
    :param optimizer_params: parameters to be used in the optimizer (OptimizerParams object)
    :return: results dictionary
    """

    # default method for adaptVQE
    if method is None:
        method = AdapVanilla()

    print('optimizer params: ', optimizer_params)

    # Initialize data structures
    iterations = {'energies': [], 'norms': [], 'f_evaluations': [], 'ansatz_size': [], 'variance': [], 'fidelity': [],
                  'coefficients': [], 'indices': []}

    # get initial coefficients
    coefficients = adapt_ansatz.parameters

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)
    print('pool size: ', len(operators_pool))

    # get n qubits to be used
    print('n_qubits: ', adapt_ansatz.n_qubits)

    # compute circuit variance
    variance = adapt_ansatz.get_energy(coefficients, hamiltonian, energy_simulator, return_std=True)[1]
    print('Initial variance: ', variance)

    # Initialize variables that are common for all the methods
    method.initialize_general_variables(hamiltonian, operators_pool, energy_threshold)

    # fill data in iterations dictionary
    iterations['variance'].append(variance)
    iterations['energies'].append(adapt_ansatz.get_energy(coefficients, hamiltonian, energy_simulator))
    iterations['coefficients'].append(coefficients)
    iterations['indices'].append(adapt_ansatz.operators.get_index(operators_pool))

    for iteration in range(max_iterations):
        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))

        # update ansatz
        try:
            adapt_ansatz = method.update_ansatz(adapt_ansatz, iterations)
        except Converged as c:
            print(c.message)
            return {'energy': iterations['energies'][-1],
                    'ansatz': adapt_ansatz,
                    'indices': iterations['indices'][-1],
                    'coefficients': iterations['coefficients'][-1],
                    'iterations': iterations,
                    'variance': iterations['variance'][-1],
                    'num_iterations': iteration}

        # run optimization
        if energy_simulator is not None:
            energy_simulator.update_model(precision=energy_threshold,
                                          variance=variance,
                                          n_coefficients=len(coefficients),
                                          n_qubits=adapt_ansatz.n_qubits)

        results_vqe = vqe(hamiltonian, adapt_ansatz, energy_simulator, energy_threshold, optimizer_params)

        print('\n{:^12}   {}'.format('coefficient', 'operator'))
        for c, op in zip(adapt_ansatz.parameters, adapt_ansatz.operators):
            if adapt_ansatz.operators.is_fermionic():
                print('{:12.5e}   {} '.format(c, get_string_from_fermionic_operator(op)))
            else:
                print('{:12.5e} {} '.format(c, str(op).replace('\n', '')))
        print()

        if reference_dm is not None:
            density_matrix = get_density_matrix(adapt_ansatz)
            fidelity = density_fidelity(reference_dm, density_matrix)
            print('fidelity: {:6.4e}'.format(fidelity))
            iterations['fidelity'].append(fidelity)

        # print iteration results
        print('Iteration energy:', results_vqe['energy'])
        print('Ansatz Indices:', adapt_ansatz.operators.get_index(operators_pool))

        # Data storage
        iterations['energies'].append(results_vqe['energy'])
        iterations['f_evaluations'].append(results_vqe['f_evaluations'])
        iterations['ansatz_size'].append(len(adapt_ansatz))
        iterations['variance'].append(variance)
        iterations['coefficients'].append(results_vqe['coefficients'])
        iterations['indices'].append(adapt_ansatz.operators.get_index(operators_pool))

        # Checking criteria convergence
        criteria_list = method.get_criteria_list_convergence()
        for criteria in criteria_list:
            try:
                criteria(iterations)
            except Converged as c:
                print(c.message)
                return {'energy': iterations['energies'][-1],
                        'ansatz': adapt_ansatz,
                        'indices': iterations['indices'][-1],
                        'coefficients': iterations['coefficients'][-1],
                        'iterations': iterations,
                        'variance': iterations['variance'][-1],
                        'num_iterations': iteration}

        # to be deprecated
        if energy_simulator is not None:
            circuit_info = energy_simulator.get_circuit_info(adapt_ansatz)
            print('Energy circuit depth: ', circuit_info['depth'])

    raise NotConvergedError({'energy': iterations['energies'][-1],
                             'ansatz': adapt_ansatz,
                             'indices': iterations['indices'][-1],
                             'coefficients': iterations['coefficients'][-1],
                             'iterations': iterations,
                             'variance': iterations['variance'][-1],
                             'num_iterations': len(adapt_ansatz)})

if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from pool import get_pool_singlet_sd
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
    from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
    from analysis import get_info

    Configuration().verbose = True

    distance = 1.5
    basis = '3-21g'
    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, distance]],
                                  ['H', [0, 0, 2 * distance]],
                                  ['H', [0, 0, 3 * distance]]],
                                basis=basis,
                                multiplicity=1,
                                charge=0,
                                description='H4')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get additional info about electronic structure properties
    # get_info(molecule, check_HF_data=False)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals

    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

    # print data
    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)
    print('n_qubits:', n_orbitals*2)

    # Get a pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
    print('hf reference', hf_reference_fock)

    # define ansatz
    adapt_ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

    # Simulator
    # from simulators.penny_simulator import PennylaneSimulator as Simulator
    # from simulators.cirq_simulator import CirqSimulator as Simulator
    from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          shots=1000)

    # Method
    from vqemulti.method.adapt_vanila import AdapVanilla
    from vqemulti.method.tetris_adapt import AdapTetris
    from vqemulti.method.genetic_adapt import GeneticAdapt
    from vqemulti.method.genetic_add_adapt import Genetic_Add_Adapt

    method_vanilla = AdapVanilla(gradient_threshold=1e-6,
                             diff_threshold=0,
                             coeff_tolerance=1e-10,
                             # gradient_simulator=simulator,
                             operator_update_number=1,
                             operator_update_max_grad=2e-2,
                             )

    method_tetris = AdapTetris(gradient_threshold=1e-6,
                               diff_threshold=0,
                               coeff_tolerance=1e-10,
                               # gradient_simulator=simulator,
                               operator_update_max_grad=0.001
                               )

    method_genetic = GeneticAdapt(gradient_threshold=1e-6,
                                  diff_threshold=0,
                                  coeff_tolerance=1e-10,
                                  gradient_simulator=None,
                                  beta=6
                                  )

    method_genetic_add = Genetic_Add_Adapt(gradient_threshold=1e-6,
                                           diff_threshold=0,
                                           coeff_tolerance=1e-10,
                                           gradient_simulator=None,
                                           beta=6,
                                           alpha=0.001
                                           )

    try:
        result = adaptVQE(hamiltonian,     # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          adapt_ansatz,
                          energy_threshold=0.0001,
                          # method=method_genetic_add,
                          max_iterations=20,
                          # energy_simulator=simulator,
                          # variance_simulator=simulator,
                          reference_dm=None,
                          # optimizer_params=OptimizerParams()
                          )
    except NotConvergedError as c:
        print('Not converged :(')
        result = c.results

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy adaptVQE: ', result['energy'])
    print('Energy FullCI: ', molecule.fci_energy)

    error = result['energy'] - molecule.fci_energy
    print('Error:', error)

    #print('Ansatz:', result['ansatz'])
    print('Coefficients:', result['coefficients'])
    print('Operator Indices:', result['indices'])
    print('Num operators: {}'.format(len(result['ansatz'])))
