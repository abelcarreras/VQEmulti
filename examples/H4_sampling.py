from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
import matplotlib.pyplot as plt
import numpy as np


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
# h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# hamiltonian analysis
max_2body = np.max(np.abs(hamiltonian.two_body_tensor))
max_1body = np.max(np.abs(hamiltonian.one_body_tensor))
print('Max valued hamiltonian terms: ', max_1body, max_2body)


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

print('\nPOOL\n---------------------------')
pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      shots=100)

# simulator = None
precision = 1e-2
print('precision', precision)

# run adaptVQE
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      #opt_qubits=True,
                      energy_threshold=precision,
                      # coeff_tolerance=precision,
                      #threshold=0.1,
                      max_iterations=5,
                      operator_update_number=1,
                      operator_update_max_grad=precision,
                      energy_simulator=simulator,
                      #gradient_simulator=simulator,
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
#print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

indices_1 = result["indices"]

energies_list = result['iterations']['energies']
error_list = np.array(energies_list) - molecule.fci_energy
norms_list = result['iterations']['norms']
n_evaluations = np.sum(np.multiply(result['iterations']['f_evaluations'],
                                   result['iterations']['ansatz_size']))

print('n_evaluations: ', n_evaluations)

# fidelity
density_matrix = get_density_matrix(result['coefficients'],
                                    result['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('\nFidelity measure: {:5.2f}'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))
simulator.print_statistics()
