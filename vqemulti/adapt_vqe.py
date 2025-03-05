from select import select

from vqemulti.energy import exact_adapt_vqe_energy
from vqemulti.utils import get_string_from_fermionic_operator
from vqemulti.pool.tools import OperatorList
from vqemulti.errors import NotConvergedError, Converged
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_adapt_vqe_energy
from vqemulti.gradient.simulation import simulate_vqe_energy_gradient
from vqemulti.gradient.exact import exact_adapt_vqe_energy_gradient
from vqemulti.optimizers import OptimizerParams
from vqemulti.method.adapt_vanila import AdapVanilla
from vqemulti import prune_adapt
import numpy as np
import scipy


def adaptVQE(hamiltonian,
             operators_pool,
             hf_reference_fock,
             energy_threshold=0.0001,
             max_iterations=50,
             coefficients=None,
             ansatz=None,
             method=AdapVanilla(),
             prune = None,
             energy_simulator=None,
             variance_simulator=None,
             reference_dm=None,
             optimizer_params=None,
             ):
    """
    Perform an adaptVQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param operators_pool: fermionic operators pool
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param max_iterations: max number of adaptVQE iterations
    :param coefficients: Initial coefficients (None if new calculation)
    :param ansatz: Initial ansatz [Should match with coefficients] (None if new calculation)
    :param method: Method used to update the ansatz
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param gradient_simulator: Simulator object used to obtain the gradient, if None do not use simulator (exact)
    :param variance_simulator: Simulator object used to obtain the variance, if None use energy_simulator
    :param reference_dm: reference density matrix (ideally from fullCI) that is used to compute the quantum fidelity
    :param optimizer_params: parameters to be used in the optimizer (OptimizerParams object)
    :return: results dictionary
    """

    # set default optimizer params
    if optimizer_params is None:
        optimizer_params = OptimizerParams()

    print('optimizer params: ', optimizer_params)

    # Initialize data structures
    iterations = {'energies': [], 'norms': [], 'f_evaluations': [], 'ansatz_size': [], 'variance': [], 'fidelity': [],
                  'coefficients': [], 'indices': []}

    # Check if initial guess
    if ansatz is None:
        ansatz = OperatorList([])

    if coefficients is None:
        coefficients = []

    assert len(coefficients) == len(ansatz)

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)
    print('pool size: ', len(operators_pool))

    # get n qubits to be used
    n_qubits = len(hf_reference_fock)
    print('n_qubits: ', n_qubits)

    # set variance simulator
    if energy_simulator is not None:
        # variance_simulator = gradient_simulator if variance_simulator is None else variance_simulator
        variance_simulator = energy_simulator if variance_simulator is None else variance_simulator

    # compute circuit variance
    if variance_simulator is not None:
        # Calculation of Hamiltonian variance
        variance = simulate_adapt_vqe_variance(coefficients, ansatz, hf_reference_fock, hamiltonian, variance_simulator)
        print('Hamiltonian Variance: ', variance)
    else:
        variance = 0
    iterations['variance'].append(variance)

    # Initialize variables that are common for all the methods
    method.initialize_general_variables(hf_reference_fock, hamiltonian, operators_pool, energy_threshold)

    # Hartree-Fock energy calculation
    if energy_simulator is None:
        iterations['energies'].append(exact_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian))
    else:
        energy_simulator.update_model(precision=energy_threshold,
                                      variance=variance,
                                      n_coefficients=len(coefficients),
                                      n_qubits=n_qubits)
        iterations['energies'].append(simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian,
                                                                energy_simulator))
    iterations['coefficients'].append(coefficients)
    iterations['indices'].append(ansatz.get_index(operators_pool))

    for iteration in range(max_iterations):

        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))

        # update ansatz
        try:
            ansatz, coefficients = method.update_ansatz(ansatz, iterations)
        except Converged as c:
            print(c.message)
            return {'energy': iterations['energies'][-1],
                    'ansatz': ansatz,
                    'indices': iterations['indices'][-1],
                    'coefficients': iterations['coefficients'][-1],
                    'iterations': iterations,
                    'variance': iterations['variance'][-1],
                    'num_iterations': iteration}

        # run optimization
        if energy_simulator is None:
            results = scipy.optimize.minimize(exact_adapt_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian),
                                              jac=exact_adapt_vqe_energy_gradient,
                                              method=optimizer_params.method,
                                              options=optimizer_params.options,
                                              tol=energy_threshold,
                                              )
        else:
            energy_simulator.update_model(precision=energy_threshold,
                                          variance=variance,
                                          n_coefficients=len(coefficients),
                                          n_qubits=n_qubits)

            results = scipy.optimize.minimize(simulate_adapt_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian, energy_simulator),
                                              jac=simulate_vqe_energy_gradient,
                                              method=optimizer_params.method,
                                              options=optimizer_params.options,
                                              tol=energy_threshold,
                                              )

        coefficients = list(results.x)
        energy = results.fun

        if prune is not None:
            prune_method = prune
            prune_method.load_values(coefficients, hf_reference_fock, hamiltonian, energy)
            energy, coefficients, ansatz = prune_method.run_pruning(ansatz)

        print('\n{:^12}   {}'.format('coefficient', 'operator'))
        for c, op in zip(coefficients, ansatz):
            if ansatz.is_fermionic():
                print('{:12.5e}   {} '.format(c, get_string_from_fermionic_operator(op)))
            else:
                print('{:12.5e} {} '.format(c, op))
        print()

        if reference_dm is not None:
            n_orb = len(hf_reference_fock)//2
            density_matrix = get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orb)
            fidelity = density_fidelity(reference_dm, density_matrix)
            print('fidelity: {:6.4e}'.format(fidelity))
            iterations['fidelity'].append(fidelity)

        # print iteration results
        print('Iteration energy:', energy)
        print('Ansatz Indices:', ansatz.get_index(operators_pool))

        # Data storage
        iterations['energies'].append(energy)
        iterations['f_evaluations'].append(results.nfev)
        iterations['ansatz_size'].append(len(coefficients))
        iterations['variance'].append(variance)
        iterations['coefficients'].append(coefficients)
        iterations['indices'].append(ansatz.get_index(operators_pool))
        print(iterations['coefficients'])

        # Checking criteria convergence
        criteria_list = method.get_criteria_list_convergence()
        for criteria in criteria_list:
            try:
                criteria(iterations)
            except Converged as c:
                print(c.message)
                return {'energy': iterations['energies'][-1],
                        'ansatz': ansatz,
                        'indices': iterations['indices'][-1],
                        'coefficients': iterations['coefficients'][-1],
                        'iterations': iterations,
                        'variance': iterations['variance'][-1],
                        'num_iterations': iteration}

        if method.gradient_simulator is not None:
            circuit_info = method.gradient_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Gradient circuit depth: ', circuit_info['depth'])

        if energy_simulator is not None:
            circuit_info = energy_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Energy circuit depth: ', circuit_info['depth'])

    raise NotConvergedError({'energy': iterations['energies'][-1],
                             'ansatz': ansatz,
                             'indices': iterations['indices'][-1],
                             'coefficients': iterations['coefficients'][-1],
                             'iterations': iterations,
                             'variance': iterations['variance'][-1],
                             'num_iterations': len(ansatz)})

if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from pool import get_pool_singlet_sd
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
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

    # Simulator
    from simulators.penny_simulator import PennylaneSimulator as Simulator
    # from simulators.cirq_simulator import CirqSimulator as Simulator
    # from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          shots=1000)

    # Method
    from vqemulti.method.adapt_vanila import AdapVanilla
    #from vqemulti.method.tetris_adapt import AdapTetris
    #from vqemulti.method.genetic_adapt import GeneticAdapt
    #from vqemulti.method.genetic_add_adapt import Genetic_Add_Adapt

    method = AdapVanilla(gradient_threshold=1e-6,
                         diff_threshold=0,
                         coeff_tolerance=1e-10,
                         gradient_simulator=None,
                         operator_update_number=1,
                         operator_update_max_grad=2e-2,
                         )
    try:
        result = adaptVQE(hamiltonian,     # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          hf_reference_fock,
                          energy_threshold=0.0001,
                          method = method,
                          max_iterations = 20,
                          energy_simulator = None,
                          variance_simulator = None,
                          reference_dm = None,
                          optimizer_params = None
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
