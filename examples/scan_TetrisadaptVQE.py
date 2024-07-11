# example of hydrogen molecule dissociation using adaptVQE method
# and Pennylane simulator (1000 shots)
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np

vqe_energies = []
energies_fullci = []
energies_hf = []
for d in np.linspace(0.3, 3, 20):

    # molecule definition
    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, d]],
                                  ['H', [0, 0, 2 * d]],
                                  ['H', [0, 0, 3 * d]]],
                                basis='3-21g',
                                multiplicity=1,
                                charge=0,
                                description='H4')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True)

    # get additional info about electronic structure properties
    # get_info(molecule, check_HF_data=False)

    # get properties from classical SCF calculation
    n_electrons = 4  # molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()

    # Choose specific pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(n_electrons=n_electrons,
                               n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

    # define simulator paramters
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          shots=1000)

    from vqemulti.method.tetris_adapt import AdapTetris

    method = AdapTetris(gradient_threshold=1e-6,
                         diff_threshold=0,
                         coeff_tolerance=1e-10,
                         gradient_simulator=None,
                         operator_update_max_grad = 0.001
                         )
    # run adaptVQE
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


    print("Final energy:", result["energy"])

    # Compare error vs FullCI calculation
    error = result["energy"] - molecule.fci_energy
    print("Error:", error)

    # run results
    print("Ansatz:", result["ansatz"])
    print("Indices:", result["indices"])
    print("Coefficients:", result["coefficients"])
    print("Num operators: {}".format(len(result["ansatz"])))

    energies_hf.append(molecule.hf_energy)
    vqe_energies.append(result["energy"])
    energies_fullci.append(molecule.fci_energy)

plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

plt.plot(np.linspace(0.3, 3, 20), energies_hf, label='HF')
plt.plot(np.linspace(0.3, 3, 20), vqe_energies, label='adaptVQE')
plt.plot(np.linspace(0.3, 3, 20), energies_fullci, label='FullCI')
plt.legend()

plt.figure()
plt.title('Difference between fullCI')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')
plt.yscale('log')

diff_fullci = np.subtract(vqe_energies, energies_fullci)
plt.plot(np.linspace(0.3, 3, 20), diff_fullci, label='adaptVQE')
plt.legend()

plt.show()