from vqemulti.energy import exact_vqe_energy, simulate_vqe_energy, get_vqe_energy, exact_vqe_energy_gradient
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.utils import get_string_from_fermionic_operator
from vqemulti.pool.tools import OperatorList
from vqemulti.errors import NotConvergedError
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.genetic_tools.mutations import add_new_excitation, delete_excitation, change_excitation
import scipy
import numpy as np
import warnings
from copy import deepcopy
import math

def geneticVQE(hamiltonian,
             operators_pool,
             hf_reference_fock,
             opt_qubits=False,
             max_iterations=50,
             add_probability = 0.6,
             delete_probability = 0.2,
             change_probability = 0.2,
             population_number = 0.8,
             cheat_param = 0.2,
             coefficients=None,
             ansatz=None,
             einstein_index = None,
             energy_simulator=None,
             gradient_simulator=None,
             coeff_tolerance=1e-10,
             energy_threshold=1e-4,
             threshold=1e-6,
             operator_update_number=1,
             operator_update_max_grad=2e-1,
             reference_dm=None,
                ):
    """
    Perform an adaptVQE calculation

    :param operators_pool: fermionic operators pool
    :param hamiltonian: hamiltonian in fermionic operators
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param max_iterations: max number of adaptVQE iterations
    :param coefficients: Initial coefficients (None if new calculation)
    :param ansatz: Initial ansatz [Should match with coefficients] (None if new calculation)
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param gradient_simulator: Simulator object used to obtain the gradient, if None do not use simulator (exact)
    :param coeff_tolerance: Set upper limit value for coefficient to be considered as zero
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param threshold: total-gradient-norm convergence threshold (in Hartree)
    :param operator_update_number: number of operators to add to the ansatz at each iteration
    :param operator_update_max_grad: max gradient relative deviation between operations that update together in one iteration
    :param reference_dm: reference density matrix (ideally from fullCI) that is used to compute the quantum fidelity
    :return: results dictionary
    """

    # Initialize data structures
    iterations = {'energies': [], 'norms': [], 'f_evaluations': [], 'ansatz_size': []}
    indices = []
    fidelities = []
    number_cnots = []

    # Check if initial guess
    if ansatz is None:
        ansatz = OperatorList([])

    if coefficients is None:
        coefficients = []

    if einstein_index is None:
        einstein_index = []

    assert len(coefficients) == len(ansatz)

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)


    if opt_qubits:
        # transform to qubit ansatz
        operators_pool = operators_pool.get_quibits_list(normalize=True)

    print('pool size: ', len(operators_pool))

    for iteration in range(max_iterations):

        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))

        #HERE I INSERT GENETIC ALGORITHM INSTEAD OF ADAPT-VQE
        fitness_vector = []
        peasants_list= []
        coefficients_list = []
        index_peasant = []
        number_peasants = int(len(operators_pool)*population_number)
        selected_already = []

        if iteration == 0:
            for i in range(number_peasants):
                new_peasant = add_new_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients,
                                                 operators_pool,
                                                 indices, cheat_param, selected_already)
                peasants_list.append(new_peasant[0])
                coefficients_list.append(new_peasant[1])
                fitness_vector.append(new_peasant[2])
                index_peasant.append(new_peasant[3])
                selected_already.append(new_peasant[4])
        else:
            for i in range(number_peasants):
                random_number= np.random.rand()
                if 0 <= random_number <= add_probability:
                    #print('will add to', ansatz)
                    new_peasant = add_new_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients, operators_pool,
                                                     indices, cheat_param, selected_already)
                    peasants_list.append(new_peasant[0])
                    coefficients_list.append(new_peasant[1])
                    fitness_vector.append(new_peasant[2])
                    index_peasant.append(new_peasant[3])
                    selected_already.append(new_peasant[4])
                    #print('added with result', new_peasant[3])

                if add_probability < random_number <= add_probability + delete_probability:
                    #print('will delete from', ansatz)
                    new_peasant = delete_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients, indices)
                    peasants_list.append(new_peasant[0])
                    coefficients_list.append(new_peasant[1])
                    fitness_vector.append(new_peasant[2])
                    index_peasant.append(new_peasant[3])
                    #print('deleted with result', new_peasant[3])

                if add_probability + delete_probability < random_number <= 1:
                    new_peasant = change_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients,operators_pool,
                                                    indices, cheat_param)
                    peasants_list.append(new_peasant[0])
                    coefficients_list.append(new_peasant[1])
                    fitness_vector.append(new_peasant[2])
                    index_peasant.append(new_peasant[3])




        max_indices = np.argsort(fitness_vector)[0]
        second_indice = np.argsort(fitness_vector)[1]

        # get Einstein from generation
        einstein = peasants_list[max_indices]
        einstein_coefficients = coefficients_list[max_indices]
        einstein_fitness = fitness_vector[max_indices]
        einstein_index = index_peasant[max_indices]
        if len(indices) <= 1:
            # Initialize the coefficient of the operator that will be newly added at 0
            coefficients = einstein_coefficients
            ansatz = einstein
            indices = einstein_index
            print('\n')
            print('\n')
            print('The Einstein of the generation is', indices)
            print('\n')
            print('\n')

        if len(indices) > 1:
            if einstein_index[-1] == einstein_index[-2]:
                print('\n')
                print('\n')
                print('REPEATED OPS, THE EINSTEIN WAS', einstein_index)
                print('eins ind -1',einstein_index[-1], einstein_index[-2])
                coefficients = coefficients_list[second_indice]
                ansatz = peasants_list[second_indice]
                indices = index_peasant[second_indice]
                print('WE Ktake the second one', indices)
                print('\n')
                print('\n')

            else:
                # Initialize the coefficient of the operator that will be newly added at 0
                coefficients = einstein_coefficients
                ansatz = einstein
                indices = einstein_index
                print('\n')
                print('\n')
                print('The Einstein of the generation is', indices)
                print('\n')
                print('\n')



        # run optimization
        if energy_simulator is None:
            results = scipy.optimize.minimize(exact_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian),
                                              jac=exact_vqe_energy_gradient,
                                              options={'gtol': energy_threshold,
                                                       'disp':  Configuration().verbose},
                                              method='BFGS',
                                              #method='COBYLA',
                                              tol=energy_threshold,
                                              #options={'rhobeg': 0.1, 'disp': Configuration().verbose}
                                              )
        else:
            opt_tolerance = energy_simulator.update_model(precision=energy_threshold,
                                                          n_coefficients=len(coefficients),
                                                          c_constant=0.4)

            results = scipy.optimize.minimize(simulate_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian, energy_simulator),
                                              method='COBYLA', # SPSA for real hardware
                                              tol=opt_tolerance,
                                              options={'disp': Configuration().verbose, 'rhobeg': 0.1})


        #print('ITERATIONS', iterations)
        #print('RESULT MIN', results)

        # get results
        coefficients = list(results.x)
        energy = results.fun

        if abs(results.x[-1]) < coeff_tolerance or abs(results.x[-1]) % math.pi < 0.001:
            ansatz = ansatz[:-1]
            indices = indices[:-1]
            coefficients = coefficients[:-1]




        print('\n{:^8}   {}'.format('coefficient', 'operator'))
        for c, op in zip(coefficients, ansatz):
            if opt_qubits:
                print('{:8.5f}   {} '.format(c, op))
            else:
                print('{:8.5f} {} '.format(c, get_string_from_fermionic_operator(op)))
        print()


        if reference_dm is not None:
            n_orb = len(hf_reference_fock)//2
            density_matrix = get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orb)
            fidelity = density_fidelity(reference_dm, density_matrix)
            fidelities.append(fidelity)
            print('fidelity',fidelity)
        
        # print iteration results
        print('Iteration energy:', energy)
        # print('Coefficients:', coefficients)
        print('Final indices selected', indices)

        iterations['energies'].append(energy)
        #iterations['norms'].append(total_norm)
        iterations['f_evaluations'].append(results.nfev)
        iterations['ansatz_size'].append(len(coefficients))
        #print(iterations['energies'])

        #energy_simulator.print_statistics()
        if gradient_simulator is not None:
            circuit_info = gradient_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Gradient circuit depth: ', circuit_info['depth'])

        if energy_simulator is not None:
            circuit_info = energy_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Energy circuit depth: ', circuit_info['depth'])



        #y = energy_simulator.print_statistics()
        #number_cnots.append(y)




        if iteration == max_iterations - 1:
            warnings.warn('finished due to max iterations reached')
            return {'energy': iterations['energies'][-1],
                    'ansatz': ansatz,
                    'indices': indices,
                    'coefficients': coefficients,
                    'iterations': iterations,
                    'number cnots': number_cnots,
                    'fidelities': fidelities}

        if 1.96327803704193 + energy < 1e-4 :
            warnings.warn('finished due to adapt ansatz reached')
            return {'energy': iterations['energies'][-1],
                    'ansatz': ansatz,
                    'indices': indices,
                    'coefficients': coefficients,
                    'iterations': iterations,
                    'number cnots': number_cnots,
                    'fidelities': fidelities}


    '''
    raise NotConvergedError({'energy': iterations['energies'][-1],
                             'ansatz': ansatz,
                             'indices': indices,
                             'coefficients': coefficients,
                             'iterations': iterations})
    '''


if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from pool import get_pool_singlet_sd
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
    from analysis import get_info

    Configuration().verbose = True

    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, 0.74]]],
                                basis='3-21g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

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
    print('n_qubits:', hamiltonian.n_qubits)

    # Get a pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
    print('hf reference', hf_reference_fock)

    # Simulator
    #from simulators.penny_simulator import PennylaneSimulator as Simulator
    # from simulators.cirq_simulator import CirqSimulator as Simulator
    from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          shots=1000)

    result = adaptVQE(hamiltonian,     # fermionic hamiltonian
                      operators_pool,  # fermionic operators
                      hf_reference_fock,
                      opt_qubits=False,
                      max_iterations=30,
                      coeff_tolerance=1e-3,
                      energy_threshold=1e-4,
                      threshold=1e-9,
                      energy_simulator=simulator,
                      gradient_simulator=simulator
                      )

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy adaptVQE: ', result['energy'])
    print('Energy FullCI: ', molecule.fci_energy)

    error = result['energy'] - molecule.fci_energy
    print('Error:', error)

    print('Ansatz:', result['ansatz'])
    print('Coefficients:', result['coefficients'])
    print('Operator Indices:', result['indices'])
    print('Num operators: {}'.format(len(result['ansatz'])))
