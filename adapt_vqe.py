from utils import get_hf_reference_in_fock_space
from energy import exact_vqe_energy, simulate_vqe_energy
from gradient import compute_gradient_vector, simulate_gradient
from openfermion.transforms import jordan_wigner
from pool_definitions import generate_jw_operator_pool
import numpy as np
import scipy


def adaptVQE(operators_pool,  # fermionic operators
             hamiltonian,     # fermionic hamiltonian
             hf_reference_fock,  # used to determine reference HF
             max_iterations=50,
             threshold=0.1,
             exact_energy=True,
             exact_gradient=True,
             trotter=True,
             trotter_steps=1,
             sample=True,
             shots=1000):

    # Initialize data structures
    iterations = {"energies": [], "norms": []}
    qubit_ansatz = []
    coefficients = []
    indices = []

    # transform to qubits hamiltonian (JW transformation)
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # transform to qubits operators (JW transformation)
    qubit_operators_pool = generate_jw_operator_pool(operators_pool)
    print("qubit pool size:", len(qubit_operators_pool))

    for iteration in range(max_iterations):

        print("\n*** Adapt Iteration {} ***\n".format(iteration+1))

        if len(qubit_ansatz) != 0:
            print('ansatz: ', qubit_ansatz)
            print('coefficients: ', coefficients)

        if exact_gradient:
            gradient_vector = compute_gradient_vector(hf_reference_fock,
                                                      qubit_hamiltonian,
                                                      qubit_ansatz,
                                                      coefficients,
                                                      qubit_operators_pool)
        else:
            gradient_vector = simulate_gradient(hf_reference_fock,
                                                qubit_hamiltonian,
                                                qubit_ansatz,
                                                coefficients,
                                                qubit_operators_pool,
                                                shots,
                                                sample)

        total_norm = np.linalg.norm(gradient_vector)
        max_index = np.argmax(gradient_vector)
        max_gradient = np.max(gradient_vector)
        max_operator = qubit_operators_pool[max_index]

        print("Total gradient norm: {}".format(total_norm))

        if total_norm < threshold:
            print("\nConvergence condition achieved!")
            result = {'energy': iterations["energies"][-1],
                      'ansatz': qubit_ansatz,
                      'indices': indices,
                      'coefficients': coefficients}

            return result, iterations

        print("Selected: {} (norm {:.6f})".format(max_index, max_gradient))

        # Initialize the coefficient of the operator that will be newly added at 0
        coefficients.append(0)
        qubit_ansatz.append(max_operator)
        indices.append(max_index)

        # run optimization
        if exact_energy:
            opt_result = scipy.optimize.minimize(exact_vqe_energy,
                                                 coefficients,
                                                 (qubit_ansatz, hf_reference_fock, qubit_hamiltonian),
                                                 method="COBYLA",
                                                 tol=None,
                                                 options={'rhobeg': 0.1, 'disp': True})
        else:
            opt_result = scipy.optimize.minimize(simulate_vqe_energy,
                                                 coefficients,
                                                 (qubit_ansatz, hf_reference_fock, qubit_hamiltonian,
                                                  shots, trotter, trotter_steps, sample),
                                                 method="COBYLA",
                                                 tol=1e-8,
                                                 options={ 'disp': True}) # 'rhobeg': 0.01)


        coefficients = list(opt_result.x)

        # Energy obtained by exact function (using optimized coefficients)
        optimized_energy = exact_vqe_energy(coefficients, qubit_ansatz, hf_reference_fock, qubit_hamiltonian)

        print("Optimized Energy:", optimized_energy)
        print("Coefficients:", coefficients)
        print("Ansatz Indices:", indices)

        iterations["energies"].append(optimized_energy)
        iterations["norms"].append(total_norm)

    raise Exception('Not converged!')


if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from pool_definitions import get_pool_singlet_sd

    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, 0.74]]],
                                basis='3-21g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    hamiltonian = molecule.get_molecular_hamiltonian()
    n_electrons = molecule.n_electrons
    n_orbitals = 2 # molecule.n_orbitals

    # Choose specific pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(electronNumber=n_electrons,
                                         orbitalNumber=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    result, iterations = adaptVQE(operators_pool,  # fermionic operators
                                  hamiltonian,     # fermionic hamiltonian
                                  hf_reference_fock,
                                  threshold=0.1,
                                  exact_energy=True,
                                  exact_gradient=True,
                                  trotter=False,
                                  sample=False,
                                  shots=100)

    print("Optimized AdaptVQE energy:", result["energy"])
    print("FullCI energy:", molecule.fci_energy)

    error = result["energy"] - molecule.fci_energy
    print("Error:", error)

    # Define chemical accuracy
    chemicalAccuracy = 1.5936e-3  # Hartree
    print("(in % of chemical accuracy: {:.3f}%)\n".format(error / chemicalAccuracy * 100))

    print("Ansatz:", result["ansatz"])
    print("Indices:", result["indices"])
    print("Coefficients:", result["coefficients"])
    print("Num operators: {}".format(len(result["ansatz"])))
