from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd, get_pool_spin_complement_gsd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.errors import NotConvergedError

vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, square_h4_mol, linear_h4_mol
#h4_molecule = tetra_h4_mol(distance=5.0, basis='3-21g')
h4_molecule = square_h4_mol(distance=3.0, basis='3-21g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=True, guess_mix=True)
print('FullCI energy result:', molecule.fci_energy)
# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 5  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)
print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
# pool = get_pool_spin_complement_gsd(n_orbitals=n_orbitals)
pool.print_compact_representation()


# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

simulator = Simulator(trotter=True,
                        trotter_steps=1,
                        # test_only=True,
                        shots=1000)

# run adaptVQE
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      #opt_qubits=True,
                      threshold=1e-7,
                      max_iterations=20,
                      # energy_simulator=simulator,
                      # gradient_simulator=simulator,
                      )

except NotConvergedError as e:
    result = e.results

print("HF energy:", molecule.hf_energy)
print("Final adaptVQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)


# Compare error vs FullCI calculation
error = result["energy"] - molecule.fci_energy
print("Error:", error)

# run results
print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

energies_list = result['iterations']['energies']
error_list = np.array(energies_list) - molecule.fci_energy
norms_list = result['iterations']['norms']

print(norms_list)

plt.plot(energies_list, label='AdaptVQE', color='blue')
plt.title('Energies')
plt.ylabel('Energy [H]')

plt.figure()
plt.title('Error')
plt.ylabel('Energy [H]')
plt.yscale('log', base=10)
plt.plot(error_list, 'o-', label='AdaptVQE', color='red')

plt.figure()
plt.title('Norm')
plt.plot(norms_list, 'o-', label='AdaptVQE', color='red')

plt.show()


for i, en in enumerate(energies_list):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list[i], norms_list[i]))