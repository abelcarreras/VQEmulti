from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
import math

vqe_energies = []
energies_fullci = []
energies_hf = []

def generate_tetrahedron_coordinates(axis_length):
    # Calculate the height of the tetrahedron
    height = math.sqrt(2/3) * axis_length

    # Calculate the coordinates of the vertices
    vertices = []
    vertices.append((0, 0, 0))
    vertices.append((axis_length, 0, 0))
    vertices.append((axis_length/2, axis_length*math.sqrt(3)/2, 0))
    vertices.append((axis_length/2, axis_length*math.sqrt(3)/6, height))

    return vertices


def tetra_h4_mol(distance, basis='sto-3g'):
    coor = generate_tetrahedron_coordinates(distance)

    mol = MolecularData(geometry=[['H', coor[0]],
                                  ['H', coor[1]],
                                  ['H', coor[2]],
                                  ['H', coor[3]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol


def square_h4_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [distance, 0, 0]],
                                  ['H', [0, distance, 0]],
                                  ['H', [distance, distance, 0]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol

def linear_h4_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, distance]],
                                  ['H', [0, 0, 2 * distance]],
                                  ['H', [0, 0, 3 * distance]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol


# run classical calculation
molecule = run_pyscf(linear_h4_mol(1.5, basis='3-21g'), run_fci=True)

# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = 4  # molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons,
                           n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

# define simulator paramters
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

# run adaptVQE
result = adaptVQE(hamiltonian,
                  pool,
                  hf_reference_fock,
                  opt_qubits=False,
                  max_iterations=30,
                  coeff_tolerance=1e-3,
                  energy_threshold=1e-4,
                  threshold=1e-9,
                  energy_simulator=simulator,  # comment this line to not use sampler simulator
                  gradient_simulator=simulator,  # comment this line to not use sampler simulator
                  )

print(' ')
print('-----------------------------------------')
print('ADAPT-VQE algorithm finished!')
print("Final energy:", result["energy"])
print('Number of function evaluations per optimization in  each adapt iteration', result['iterations']['f_evaluations'])
counts = result['iterations']['f_evaluations']
print('Total number of function evaluatoins in the whole algorithm', sum(counts))
print("Indices of the operators included in the ansatz:", result["indices"])
print("Total number operators added in the ansatz: {}".format(len(result["ansatz"])))



energy_iteration = result['iterations']['energies']
coefficients = result["coefficients"]
norms = result['iterations']['norms']


print(' ')
print('ENERGIES PER ITERATION')
for i in range(len(energy_iteration)):
    print(energy_iteration[i])

print('COEFFICIENTS')
for i in range(len(coefficients)):
    print(coefficients[i])

print('TOTAL GRADIENTS PER ITERATION')
for i in range(len(norms)):
    print(norms[i])