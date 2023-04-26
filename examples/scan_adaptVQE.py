from utils import get_hf_reference_in_fock_space
from pool_definitions import get_pool_singlet_sd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adapt_vqe import adaptVQE
from analysis import get_info
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_reduced_hamiltonian

vqe_energies = []
energies_fullci = []
for d in np.linspace(0.3, 3, 20):

    # molecule definition
    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, d]]],
                                basis='6-31g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True)

    # get additional info about electronic structure properties
    get_info(molecule, check_HF_data=False)

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True)

    # get properties from classical SCF calculation
    n_electrons = 2  # molecule.n_electrons
    n_orbitals = 2  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()
    generate_reduced_hamiltonian(hamiltonian, n_orbitals)

    # Choose specific pool of operators for adapt-VQE
    pool = get_pool_singlet_sd(electronNumber=n_electrons,
                               orbitalNumber=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    # run adaptVQE
    result, iterations = adaptVQE(pool,
                                  hamiltonian,
                                  hf_reference_fock,
                                  threshold=0.1, # in Hartree
                                  exact_energy=True,
                                  exact_gradient=True,
                                  test_only=True,
                                  )

    print("Final energy:", result["energy"])

    # Compare error vs FullCI calculation
    error = result["energy"] - molecule.fci_energy
    print("Error:", error)

    # Error respect to chemical accuracy
    chemicalAccuracy = 1.5936e-3
    print("(in % of chemical accuracy: {:.3f}%)\n".format(error/chemicalAccuracy*100))

    # run results
    print("Ansatz:", result["ansatz"])
    print("Indices:", result["indices"])
    print("Coefficients:", result["coefficients"])
    print("Num operators: {}".format(len(result["ansatz"])))

    vqe_energies.append(result["energy"])
    energies_fullci.append(molecule.fci_energy)

plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

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