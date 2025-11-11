from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals

#from shot_models import get_evaluation_shots_model_simple as shots_model_energy
from shot_models import get_evaluation_shots_model_3 as shots_model_energy


import matplotlib.pyplot as plt
import numpy as np
import time


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
#h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
#h4_molecule = square_h4_mol(distance=3.0, basis='3-21g')
h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

distance = 3.0
from openfermion import MolecularData
h4_molecule_ = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [distance, 0, 0]],
                                      ['H', [0, distance, 0]],
                                      ['H', [distance, distance, 0]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H4')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False, verbose=True)

# symmetrize molecular orbitals (MolecularData object)
sym_orbitals = symmetrize_molecular_orbitals(molecule, 'c2h', skip=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

max_2body = np.max(np.abs(hamiltonian.two_body_tensor))
max_1body = np.max(np.abs(hamiltonian.one_body_tensor))
print('Max valued hamiltonian terms: ', max_1body, max_2body)


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool = get_symmetry_reduced_pool(pool, sym_orbitals) # using symmetry

print('\nPOOL\n---------------------------')
pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=False,
                      shots=100
                      )

simulator.set_shots_model(shots_model_energy)

simulator_grad = Simulator(trotter=True,
                           trotter_steps=1,
                           test_only=True)

# simulator_grad = simulator

# simulator = None
precision = 1e-2
print('precision', precision)
st = time.time()

# define optimizer
opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': precision})
opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})
opt_adam = OptimizerParams(method=adam, options={'learning_rate': 0.1, 'gtol': 1e-1})


# run adaptVQE
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      # opt_qubits=True,
                      energy_threshold=precision,
                      # coeff_tolerance=precision,
                      # threshold=0.1,
                      max_iterations=20,
                      operator_update_number=1,
                      operator_update_max_grad=precision,
                      energy_simulator=simulator,
                      gradient_simulator=simulator_grad,
                      optimizer_params=opt_bfgs,
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

n_evaluations = np.sum(result['iterations']['f_evaluations'])

n_evaluations_size = np.sum(np.multiply(result['iterations']['f_evaluations'],
                                        result['iterations']['ansatz_size']))

print('n_evaluations: ', n_evaluations)
print('n_evaluations x size: ', n_evaluations_size)

# fidelity
density_matrix = get_density_matrix(result['coefficients'],
                                    result['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('\nFidelity measure: {:5.2f}'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))
simulator.print_statistics()

print('final data')
for i, en in enumerate(energies_list):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list[i], norms_list[i]))


if False:
    plt.figure(0)
    plt.plot(energies_list, 'o-', label='AdaptVQE', color='red')
    plt.axhline(y=molecule.fci_energy, color='blue', linestyle='--')
    plt.title('Energies')
    plt.ylabel('Energy [H]')

    plt.figure(1)
    plt.title('Error')
    plt.ylabel('Energy [H]')
    plt.yscale('log', base=10)
    plt.plot(error_list, 'o-', label='AdaptVQE', color='red')

    plt.figure(2)
    plt.title('Norm')
    plt.plot(norms_list, 'o-', label='AdaptVQE', color='red')


    plt.show()
