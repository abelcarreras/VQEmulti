from utils import get_hf_reference_in_fock_space
from energy import exact_vqe_energy, simulate_vqe_energy
from gradient import compute_gradient_vector, simulate_gradient
from openfermion.transforms import jordan_wigner
from pool_definitions import generate_jw_operator_pool
from pool_definitions import OperatorList
from openfermion import QubitOperator, FermionOperator
import numpy as np
import scipy


def adaptVQE(operators_pool,
             hamiltonian,
             hf_reference_fock,
             opt_qubits=True,
             max_iterations=50,
             threshold=0.1,
             exact_energy=True,
             exact_gradient=True,
             trotter=False,
             trotter_steps=2,
             test_only=False,
             shots=1000):
    """
    Perform a adapt VQE calculation

    :param operators_pool: fermionic operators pool
    :param hamiltonian: hamiltonian in fermionic operators
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param max_iterations: max adaptVQE iterations
    :param threshold: convergence threshold (in Hartree)
    :param exact_energy: Set True to compute energy analyticaly, set False to simulate
    :param exact_gradient: Set True to compute gradients analyticaly, set False to simulate
    :param trotter: Trotterize ansatz operators
    :param trotter_steps: number of trotter steps (only used if trotter=True)
    :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
    :param shots: number of samples to perform in the simulation
    :return: results dictionary
    """

    # Initialize data structures
    iterations = {'energies': [], 'norms': []}
    ansatz = OperatorList([])
    coefficients = []
    indices = []

    # transform to qubits hamiltonian (JW transformation)
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # define operatorList from pool
    operators_pool = OperatorList(operators_pool)

    if opt_qubits:
        # transform to qubit ansatz using JW transformation
        operators_pool = 1j * operators_pool.get_quibits_list(normalize=True)

    print('qubit pool size:', len(operators_pool))

    for iteration in range(max_iterations):

        print('\n*** Adapt Iteration {} ***\n'.format(iteration+1))

        if len(ansatz) != 0:
            print('ansatz: ', ansatz)
            print('coefficients: ', coefficients)

        if exact_gradient:
            gradient_vector = compute_gradient_vector(hf_reference_fock,
                                                      qubit_hamiltonian,
                                                      ansatz,
                                                      coefficients,
                                                      operators_pool)
        else:
            gradient_vector = simulate_gradient(hf_reference_fock,
                                                qubit_hamiltonian,
                                                ansatz,
                                                coefficients,
                                                operators_pool,
                                                shots,
                                                test_only)

        total_norm = np.linalg.norm(gradient_vector)
        max_index = np.argmax(gradient_vector)
        max_gradient = np.max(gradient_vector)
        max_operator = operators_pool[max_index]

        print("Total gradient norm: {}".format(total_norm))

        if total_norm < threshold:
            print("\nConvergence condition achieved!")
            result = {'energy': iterations["energies"][-1],
                      'ansatz': ansatz,
                      'indices': indices,
                      'coefficients': coefficients}

            return result, iterations

        print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

        # Initialize the coefficient of the operator that will be newly added at 0
        coefficients.append(0)
        ansatz.append(max_operator)
        indices.append(max_index)

        # run optimization
        if exact_energy:
            opt_result = scipy.optimize.minimize(exact_vqe_energy,
                                                 coefficients,
                                                 (ansatz, hf_reference_fock, qubit_hamiltonian),
                                                 method='COBYLA',
                                                 tol=None,
                                                 options={'rhobeg': 0.1, 'disp': True})
        else:
            opt_result = scipy.optimize.minimize(simulate_vqe_energy,
                                                 coefficients,
                                                 (ansatz, hf_reference_fock, qubit_hamiltonian,
                                                  shots, trotter, trotter_steps, test_only),
                                                 method='COBYLA',
                                                 tol=1e-8,
                                                 options={'disp': True}) # 'rhobeg': 0.01)

        energy_exact = exact_vqe_energy(opt_result.x, ansatz, hf_reference_fock, qubit_hamiltonian)
        energy_sim_test = simulate_vqe_energy(opt_result.x, ansatz, hf_reference_fock, qubit_hamiltonian, shots,
                                              False, trotter_steps, True)

        assert abs(energy_exact - energy_sim_test) < 1e-8

        coefficients = list(opt_result.x)
        optimized_energy = opt_result.fun

        print('Optimized Energy:', optimized_energy)
        print('Coefficients:', coefficients)
        print('Ansatz Indices:', indices)

        iterations['energies'].append(optimized_energy)
        iterations['norms'].append(total_norm)

    raise Exception('Not converged!')


if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from pool_definitions import get_pool_singlet_sd
    from utils import generate_reduced_hamiltonian
    from analysis import get_info

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
    n_orbitals = 2  # molecule.n_orbitals

    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

    # print data
    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)
    print('n_qubits:', hamiltonian.n_qubits)

    # Choose specific pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(electronNumber=n_electrons,
                                         orbitalNumber=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    result, iterations = adaptVQE(operators_pool,  # fermionic operators
                                  hamiltonian,  # fermionic hamiltonian
                                  hf_reference_fock,
                                  threshold=0.1,
                                  opt_qubits=True,
                                  exact_energy=True,
                                  exact_gradient=True,
                                  trotter=False,
                                  test_only=True,
                                  shots=10000)

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy adaptVQE: ', result['energy'])
    print('Energy FullCI: ', molecule.fci_energy)

    error = result['energy'] - molecule.fci_energy
    print('Error:', error)

    print('Ansatz:', result['ansatz'])
    print('Indices:', result['indices'])
    print('Coefficients:', result['coefficients'])
    print('Num operators: {}'.format(len(result['ansatz'])))
