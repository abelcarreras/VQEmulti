# example of hydrogen molecule dissociation using adaptVQE method
# and Pennylane simulator (1000 shots)
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.errors import NotConvergedError
from vqemulti.analysis import get_info
from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
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
                                          ['H', [0, 0, d]]],
                                basis='6-31g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True)

    # get additional info about electronic structure properties
    # get_info(molecule, check_HF_data=False)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 2  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()

    # Choose specific pool of operators for adapt-VQE
    operators_pool = get_pool_singlet_sd(n_electrons=n_electrons,
                               n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    # build initial ansatz
    ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

    # define simulator paramters
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          shots=1000)

    from vqemulti.method.adapt_vanila import AdapVanilla

    method = AdapVanilla(gradient_threshold=1e-6,
                         diff_threshold=0,
                         coeff_tolerance=1e-10,
                         gradient_simulator=None,
                         operator_update_number=1,
                         operator_update_max_grad=2e-2,
                         )

    # run adaptVQE
    try:
        result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          ansatz,
                          energy_threshold=0.0001,
                          method=method,
                          max_iterations=20,
                          energy_simulator=None,
                          reference_dm=None,
                          optimizer_params=None
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