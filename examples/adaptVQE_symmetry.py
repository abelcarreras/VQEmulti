# example of He2 calculation using adaptVQE method
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.optimizers import OptimizerParams
from vqemulti import NotConvergedError
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt


def shot_model_1(dict_parameters):
    import numpy as np

    s_prec = 1
    energy_threshold = dict_parameters['precision']/s_prec
    variance = dict_parameters['variance']
    n_coefficients = dict_parameters['n_coefficients']
    n_coefficients = 1 if n_coefficients < 1 else n_coefficients

    n_shots_ave = variance * n_coefficients ** 2 * np.pi ** 2 / (100 * energy_threshold ** 2)
    n_shots_dev = variance * n_coefficients ** (2 / 3) / (energy_threshold ** 2)

    n_shots = int(max(n_shots_ave, n_shots_dev))

    tolerance = energy_threshold* min(1/n_coefficients**(1/3), 10 /(np.pi * n_coefficients))

    return tolerance, n_shots


def shot_model(dict_parameters):

    s_prec = 1
    precision = dict_parameters['precision']/s_prec
    n_coefficients = dict_parameters['n_coefficients']
    variance = dict_parameters['variance']

    #s_opt = precision / n_coefficients**0.85
    #s_opt = precision / (n_coefficients**(0.33)) # sum() + error

    s_opt = precision / (0.42 * n_coefficients + n_coefficients**(0.33)) # sum() + error


    n_shots = int(variance / s_opt ** 2)
    n_shots = max(1, n_shots)

    return dict_parameters['precision'], n_shots


# general configuration
Configuration().verbose = 1

# molecule definition
distance = 3.0
h4_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [distance, 0, 0]],
                                      ['H', [0, distance, 0]],
                                      ['H', [distance, distance, 0]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H4')

#from generate_mol import square_h4_mol
#h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, verbose=True, nat_orb=False)

# symmetrize molecular orbitals (MolecularData object)
sym_orbitals = symmetrize_molecular_orbitals(molecule, 'c2h', skip=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()

# store/load from disk
from vqemulti.utils import store_hamiltonian, load_hamiltonian
# store_hamiltonian(hamiltonian, file='hamiltonian.npz')
# hamiltonian = load_hamiltonian(file='hamiltonian.npz')

hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)  # using active space

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool = get_symmetry_reduced_pool(pool, sym_orbitals) # using symmetry

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# define simulator
simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)
simulator.set_shots_model(shot_model)

# define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})
opt_cobyla_mod = OptimizerParams(method=cobyla_mod, options={'rhobeg': 0.1})

# define custom optimizers
opt_rmsprop = OptimizerParams(method=rmsprop, options={'learning_rate': 0.01, 'gtol': 1e-2})
opt_sgd = OptimizerParams(method=sgd, options={'learning_rate': 0.01, 'gtol': 1e-2})
opt_adam = OptimizerParams(method=adam, options={'learning_rate': 0.1, 'gtol': 1e-1})

try:
    # run adaptVQE
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      energy_simulator=simulator,
                      # gradient_simulator=simulator,
                      # variance_simulator=simulator_variance,
                      energy_threshold=1e-2,
                      max_iterations=15,  # maximum number of interations
                      optimizer_params=opt_cobyla_mod  # optimizer parameters
                      )

except NotConvergedError as e:
    result = e.results

print('Final results\n-----------')
print('HF energy:', molecule.hf_energy)
print('Final adaptVQE energy:', result["energy"])
print('FullCI energy: {:.8f}'.format(molecule.fci_energy))

# Compare error vs FullCI calculation
error = result['energy'] - molecule.fci_energy
print('Error (with respect to FullCI):', error)

# run results
# print('Ansatz:', result['ansatz'])
print('Indices:', result['indices'])
print('Coefficients:', result['coefficients'])
print('Num operators: {}'.format(len(result['ansatz'])))

# print circuit statistics
simulator.print_statistics()
# simulator.print_circuits()

# plot results
plt.plot(result['iterations']['energies'])
plt.title('Energy')
plt.ylabel('Hartree')

plt.figure()

plt.plot(result['iterations']['norms'])
plt.title('Gradient')
plt.ylabel('Hartree')
plt.show()
