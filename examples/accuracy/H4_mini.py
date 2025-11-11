import numpy as np

from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
from vqemulti.energy.simulation import simulate_adapt_vqe_energy, simulate_adapt_vqe_energy_square
from vqemulti.pool.tools import OperatorList


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True
Configuration().mapping = 'jw'

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol, h2_mol
# h_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
# h_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
#h_molecule = square_h4_mol(distance=3.0, basis='sto-3g')
h_molecule = h2_mol(distance=1.0, basis='3-21g')

# run classical calculation
molecule = run_pyscf(h_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 1 # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

#print('\nPOOL\n---------------------------')
#pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

# energy evaluation
simulator = Simulator(trotter=False, trotter_steps=1, test_only=True)

n_samples = 100
indices = []
coefficients = []

ansatz = OperatorList([pool[i] for i in indices])

energy = simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
energy2 = simulate_adapt_vqe_energy_square(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
h_variance = energy2 - energy**2

print("HF energy:", molecule.hf_energy)
print("Single point adaptVQE energy:", energy)
print("FullCI energy:", molecule.fci_energy)
print('Variance H: ', h_variance)
print('='*50 + '\n')

variance_list = []
shot_list = range(10, 100, 20)
for n_shot in shot_list:

    n_shot = int(n_shot)

    print('N_Shots: {}\n'.format(n_shot))
    # energy evaluation
    simulator = Simulator(trotter=False, trotter_steps=1, test_only=False, shots=n_shot)

    energy_list = []
    for i in range(n_samples):
        energy = simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
        print('Sample {}: {}'.format(i, energy))
        energy_list.append(energy)

    print('\nAverage: {}'.format(np.average(energy_list)))
    print('Variance: {}'.format(np.var(energy_list)))
    print('deviation: {}'.format(np.std(energy_list)))
    print( '-'*50 + '\n')

    variance_list.append(np.var(energy_list))

import matplotlib.pyplot as plt
plt.title('Variance analysis')
plt.xlabel('Number of shots')
plt.ylabel('Variance')
plt.plot(shot_list, variance_list)
plt.show()


