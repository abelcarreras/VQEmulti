from vqemulti.energy import exact_vqe_energy, simulate_vqe_energy, get_vqe_energy, exact_vqe_energy_gradient
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.utils import get_string_from_fermionic_operator
from vqemulti.pool.tools import OperatorList
from vqemulti.errors import NotConvergedError
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity
import scipy
import numpy as np
import warnings


def adaptVQE(hamiltonian,
             operators_pool,
             hf_reference_fock,
             opt_qubits=False,
             max_iterations=50,
             coefficients=None,
             ansatz=None,
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

    # Check if initial guess
    if ansatz is None:
        ansatz = OperatorList([])

    if coefficients is None:
        coefficients = []

    assert len(coefficients) == len(ansatz)

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)

    if opt_qubits:
        # transform to qubit ansatz
        operators_pool = operators_pool.get_quibits_list(normalize=True)

    print('pool size: ', len(operators_pool))

    for iteration in range(max_iterations):

        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))


        if gradient_simulator is None:
            gradient_vector = compute_gradient_vector(hf_reference_fock,
                                                      hamiltonian,
                                                      ansatz,
                                                      coefficients,
                                                      operators_pool)
        else:
            gradient_simulator.update_model(precision=energy_threshold,
                                            n_coefficients=len(coefficients),
                                            c_constant=0.4)

            gradient_vector = simulate_gradient(hf_reference_fock,
                                                hamiltonian,
                                                ansatz,
                                                coefficients,
                                                operators_pool,
                                                gradient_simulator)

        total_norm = np.linalg.norm(gradient_vector)

        print("\nTotal gradient norm: {:12.6f}".format(total_norm))
        '''
        if reference_dm is not None:
            n_orb = len(hf_reference_fock)//2
            density_matrix = get_density_matrix(coefficients, ansatz, hf_reference_fock, n_orb)
            fidelity = density_fidelity(reference_dm, density_matrix)
            fidelities.append(fidelity)
            print('fidelity: {:5.2f}'.format(fidelity))
        '''
        
        if total_norm < threshold:
            if len(iterations['energies']) > 0:
                energy = iterations['energies'][-1]
            else:
                energy = get_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, energy_simulator)

            print("\nConverge archived due to gradient norm threshold")
            result = {'energy': energy,
                      'ansatz': ansatz,
                      'indices': indices,
                      'coefficients': coefficients,
                      'iterations': iterations,
                      'fidelities': fidelities}

            return result

        # primary selection of operators
        max_indices = np.argsort(gradient_vector)[-operator_update_number:][::-1]

        # refine selection to ensure all operators are relevant
        while True:
            max_gradients = np.array(gradient_vector)[max_indices]
            max_dev = np.max(np.std(max_gradients))
            if max_dev/np.max(max_gradients) > operator_update_max_grad:
                max_indices = max_indices[:-1]
            else:
                break

        # get gradients/operators update list
        max_gradients = np.array(gradient_vector)[max_indices]
        max_operators = np.array(operators_pool)[max_indices]

        for max_index, max_gradient in zip(max_indices, max_gradients):
            print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

        # check if repeated operator
        repeat_operator = len(max_indices) == len(indices[-len(max_indices):]) and \
                          np.all(np.array(max_indices) == np.array(indices[-len(max_indices):]))

        # if repeat operator finish adaptVQE
        if repeat_operator:
            print('Converge archived due to repeated operator')
            energy = iterations['energies'][-1]
            return {'energy': energy,
                    'ansatz': ansatz,
                    'indices': indices,
                    'coefficients': coefficients,
                    'iterations': iterations,
                    'fidelities': fidelities}

        # Initialize the coefficient of the operator that will be newly added at 0
        for max_index, max_operator in zip(max_indices, max_operators):
            coefficients.append(0)
            ansatz.append(max_operator)
            indices.append(max_index)

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

        # check if last coefficient is zero (likely to happen in exact optimizations)
        if abs(results.x[-1]) < coeff_tolerance:
            print('Converge archived due to zero valued coefficient')
            n_operators = len(max_indices)
            return {'energy': results.fun,
                    'ansatz': ansatz[:-n_operators],
                    'indices': indices[:-n_operators],
                    'coefficients': coefficients[:-n_operators],
                    'iterations': iterations,
                    'fidelities': fidelities}

        # check if last iteration energy is better (likely to happen in sampled optimizations)
        diff_threshold = 0
        if len(iterations['energies']) > 0 and iterations['energies'][-1] - results.fun < diff_threshold:

            print('Converge archived due to not energy improvement')
            n_operators = len(max_indices)
            return {'energy': iterations['energies'][-1],
                    'ansatz': ansatz[:-n_operators],
                    'indices': indices[:-n_operators],
                    'coefficients': coefficients[:-n_operators],
                    'iterations': iterations}

        # get results
        coefficients = list(results.x)
        energy = results.fun

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
            print('fidelity: {:5.2f}'.format(fidelity))
        
        # print iteration results
        print('Iteration energy:', energy)
        # print('Coefficients:', coefficients)
        print('Ansatz Indices:', indices)

        iterations['energies'].append(energy)
        iterations['norms'].append(total_norm)
        iterations['f_evaluations'].append(results.nfev)
        iterations['ansatz_size'].append(len(coefficients))

        if gradient_simulator is not None:
            circuit_info = gradient_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Gradient circuit depth: ', circuit_info['depth'])

        if energy_simulator is not None:
            circuit_info = energy_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Energy circuit depth: ', circuit_info['depth'])

        if iteration == max_iterations - 1:
            warnings.warn('finished due to max iterations reached')
            return {'energy': iterations['energies'][-1],
                    'ansatz': ansatz,
                    'indices': indices,
                    'coefficients': coefficients,
                    'iterations': iterations,
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
