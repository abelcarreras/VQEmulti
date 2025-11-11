from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.density import get_density_matrix, density_fidelity
import matplotlib.pyplot as plt
import numpy as np
import time


Configuration().verbose = False

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
#h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 5  # molecule.n_orbitals
hamiltonian_full = molecule.get_molecular_hamiltonian()


print('\nPOOL\n---------------------------')
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool.print_compact_representation()

# Initial data
print('N electrons', n_electrons)
print("HF energy:", molecule.hf_energy)
print("FullCI energy:", molecule.fci_energy)

# simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=100)


# Get Hartree Fock reference in Fock space
coefficients = None
precision = 1e-1

# print info
print('N Orbitals', n_orbitals)
print('precision', precision)

hamiltonian = generate_reduced_hamiltonian(hamiltonian_full, n_orbitals)
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

st = time.time()

try:
    result = vqe(hamiltonian,
                 pool,
                 hf_reference_fock,
                 coefficients=coefficients,
                 energy_threshold=precision,
                 energy_simulator=simulator,
    )

except NotConvergedError as e:
    result = e.results

et = time.time()

print("VQE energy:", result["energy"])
print('n_evaluations: ', result["f_evaluations"])
print('n_coefficients: ', len(result["coefficients"]))
print("Coefficients:", result["coefficients"])

# fidelity
density_matrix = get_density_matrix(result['coefficients'],
                                    result['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('Fidelity measure: {:5.2f}'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))

print()
print('='*40)

print("HF energy:", molecule.hf_energy)
print("Final VQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)

error = result["energy"] - molecule.fci_energy
print("Error:", error)
print('total time: ', et - st, 's')
# Compare error vs FullCI calculation

# run results
#print("Ansatz:", result["ansatz"])

