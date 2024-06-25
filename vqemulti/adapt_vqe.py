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
import scipy


def adaptVQE(hamiltonian,
             operators_pool,
             hf_reference_fock,
             opt_qubits=False,
             max_iterations=50,
             coefficients=None,
             ansatz=None,
             method = AdapVanilla,
             energy_simulator=None,
             gradient_simulator=None,
             variance_simulator=None,
             coeff_tolerance=1e-10,
             energy_threshold=1e-4,
             gradient_threshold=1e-6,
             diff_threshold = 0,
             operator_update_number=1,
             operator_update_max_grad=2e-1,
             reference_dm=None,
             optimizer_params=None,
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
    :param variance_simulator: Simulator object used to obtain the variance, if None use energy_simulator
    :param coeff_tolerance: Set upper limit value for coefficient to be considered as zero
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :param gradient_threshold: total-gradient-norm convergence threshold (in Hartree)
    :param operator_update_number: number of operators to add to the ansatz at each iteration
    :param operator_update_max_grad: max gradient relative deviation between operations that update together in one iteration
    :param reference_dm: reference density matrix (ideally from fullCI) that is used to compute the quantum fidelity
    :param optimizer_params: parameters to be used in the optimizer (OptimizerParams object)
    :return: results dictionary
    """

    # set default optimizer params
    if optimizer_params is None:
        optimizer_params = OptimizerParams()

    print('optimizer params: ', optimizer_params)

    # Initialize data structures
    iterations = {'energies': [], 'norms': [], 'f_evaluations': [], 'ansatz_size': [], 'variance': [], 'fidelity': []}

    # Check if initial guess
    if ansatz is None:
        ansatz = OperatorList([])

    if coefficients is None:
        coefficients = []

    assert len(coefficients) == len(ansatz)

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)

    print('pool size: ', len(operators_pool))

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


    for iteration in range(max_iterations):


        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))
        method_initialization = method(energy_threshold, gradient_threshold, operator_update_number, operator_update_max_grad,
                                        gradient_simulator, diff_threshold,  hf_reference_fock, hamiltonian, ansatz, coefficients,
                                        operators_pool, variance, iterations, energy_simulator)
        try:
            ansatz, coefficients = method_initialization.update_ansatz()
        except Converged as c:
            print(c.message)
            return {'energy': c.energy,
                    'ansatz':c.ansatz,
                    'indices': c.indices,
                    'coefficients': c.coefficients,
                    'iterations': c.iterations,
                    'variance': c.variance}

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
                                          n_qubits=hamiltonian.n_qubits)

            results = scipy.optimize.minimize(simulate_adapt_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian, energy_simulator),
                                              jac=simulate_vqe_energy_gradient,
                                              method=optimizer_params.method,
                                              options=optimizer_params.options,
                                              tol=energy_threshold,
                                              )

        # get results
        coefficients = list(results.x)
        energy = results.fun

        params_check_convergence = {'results_optimization': results,
                                    'coeff_tolerance': coeff_tolerance,
                                    'diff_threshold': diff_threshold}

        method_initialization.check_convergence(params_check_convergence)


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
            print('fidelity: {:6.3f}'.format(fidelity))
            iterations['fidelity'].append(fidelity)

        # print iteration results
        print('Iteration energy:', energy)
        # print('Coefficients:', coefficients)
        print('Ansatz Indices:', ansatz.get_index(operators_pool))

        iterations['energies'].append(energy)
        iterations['f_evaluations'].append(results.nfev)
        iterations['ansatz_size'].append(len(coefficients))
        iterations['variance'].append(variance)

        if gradient_simulator is not None:
            circuit_info = gradient_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Gradient circuit depth: ', circuit_info['depth'])

        if energy_simulator is not None:
            circuit_info = energy_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
            print('Energy circuit depth: ', circuit_info['depth'])

    raise NotConvergedError({'energy': iterations['energies'][-1],
                             'ansatz': ansatz,
                             'indices': ansatz.get_index(operators_pool),
                             'coefficients': coefficients,
                             'iterations': iterations,
                             'variance': variance})


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
    print('n_qubits:', hamiltonian.n_qubits)

    # Get a pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
    print('hf reference', hf_reference_fock)

    # Simulator
    from simulators.penny_simulator import PennylaneSimulator as Simulator
    # from simulators.cirq_simulator import CirqSimulator as Simulator
    # from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          shots=1000)

    result = adaptVQE(hamiltonian,     # fermionic hamiltonian
                      operators_pool,  # fermionic operators
                      hf_reference_fock,
                      energy_threshold=0.1,
                      # opt_qubits=True,
                      # energy_simulator=simulator,
                      # gradient_simulator=simulator
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
