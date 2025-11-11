from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd, get_pool_singlet_gsd, get_pool_spin_complement_gsd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 1
config.mapping = 'jw'


def get_molecule(filename):
    """
    read molecule from xyz file

    :param filename: XYZ file
    :return: pyscf molecule
    """

    with open(filename, 'r') as f:
        lines = f.readlines()[2:]

    symbols = []
    coordinates = []

    for line in lines:
        symbols.append(line.split()[0])
        coordinates.append(line.split()[1:])

    coordinates = np.array(coordinates, dtype=float)

    geometry = []
    for s, c in zip(symbols, coordinates):
        geometry.append([s, [c[0], c[1], c[2]]])

    mol = MolecularData(geometry=geometry,
                        basis='sto-3g',
                        multiplicity=1,
                        charge=0,
                        description='molecule')

    return mol


# molecule definition
h4 = get_molecule('h4.xyz')

# run classical calculation
molecule = run_pyscf(h4, run_fci=True, nat_orb=True, guess_mix=False, verbose=True, frozen_core=0)
print('FullCI energy: : ', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()

# matrix = get_sparse_operator(hamiltonian).toarray()
# print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])

print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
# pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
# pool = get_pool_singlet_gsd(n_orbitals=n_orbitals)
# pool = get_pool_spin_complement_gsd(n_orbitals=n_orbitals)

sym_orbitals = symmetrize_molecular_orbitals(molecule, 'ci', skip=True)
# pool = get_symmetry_reduced_pool(pool, sym_orbitals, threshold=0.5)  # using symmetry
# pool = pool.get_quibits_list(normalize=True)

from vqemulti.pool import get_pool_qubit_sd, get_pool_qubit_gsd
pool = get_pool_qubit_gsd(n_orbitals=n_orbitals)
# pool = get_pool_qubit_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
print('len pool', len(pool))

from vqemulti.symmetry import get_pauli_symmetry_reduced_pool
# pool = get_pauli_symmetry_reduced_pool(pool)
print('len pool reduced', len(pool))

print('\nADAPT_VQE POOL\n---------------------------')
pool.print_compact_representation()


# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)

from vqemulti.utils import reorder_qubits
# hamiltonian = get_fermion_operator(hamiltonian)
# hamiltonian, hf_reference_fock, pool = reorder_qubits([0, 1, 2, 3], hamiltonian, hf_reference_fock, pool)

# get order
from vqemulti.utils import build_interaction_matrix, get_clustering_order
#interaction_matrix = build_interaction_matrix(hamiltonian, show_plot=True)
#order = get_clustering_order(interaction_matrix, show_plot=True)
#print('clustering order: ', order)

import itertools
all_permutations = itertools.permutations(range(n_orbitals))
print('N permutations: ', len(list(all_permutations)))

energy_lists = []
orbital_orders = []
for orbital_order in itertools.permutations(range(n_orbitals)):
    print('order: ', orbital_order)
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True,
                          shots=1024,
                          )

    precision = 1e-6
    print('precision', precision)

    # define optimizer
    opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': precision})
    opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1, 'maxiter': 2000})
    opt_adam = OptimizerParams(method=adam, options={'learning_rate': 0.1, 'gtol': 1e-1})
    # cobyla modified
    opt_cobyla_mod = OptimizerParams(method=cobyla_mod, options={'rhobeg': 0.5,
                                                                 'n_guess': 50,
                                                                 'guess_range': np.pi,
                                                                  'gtol': 1e-6,
                                                                 })
    # define update method
    from vqemulti.method.adapt_vanila import AdapVanilla

    adapt_method = AdapVanilla(operator_update_number=1,
                               operator_update_max_grad=precision,
                               gradient_threshold=1e-8,
                               # coeff_tolerance=precision*0.1,
                               # diff_threshold=0,
                               # gradient_simulator=simulator_grad,
                               diff_threshold=0,
                               min_iterations=5
                )

    from vqemulti.utils import reorder_qubits
    # hamiltonian_ro, hf_reference_fock_ro, pool_ro = reorder_qubits(orbital_order, hamiltonian, hf_reference_fock, pool)
    hamiltonian_ro, hf_reference_fock_ro = reorder_qubits(orbital_order, hamiltonian, hf_reference_fock)


    # run adaptVQE
    try:
        result = adaptVQE(hamiltonian_ro,
                          pool,
                          hf_reference_fock_ro,
                          energy_threshold=precision,
                          max_iterations=8,
                          # energy_simulator=simulator,
                          optimizer_params=opt_bfgs,
                          method=adapt_method
                          )

    except NotConvergedError as e:
        result = e.results

    #print('simulation energy')
    #simulator.print_statistics()

    print(result['iterations']['energies'])
    energy_lists.append(result['iterations']['energies'])
    orbital_orders.append(orbital_order)


print("HF energy:", molecule.hf_energy)
print("Final adaptVQE energy:", result["energy"])
#print("FullCI energy:", molecule.fci_energy)
print("FCI energy:", molecule.fci_energy)
print("Error (respect to FCI): {:10.3e}".format(result["energy"] - molecule.fci_energy))

print(result)

for order, energy_list in zip(orbital_orders, energy_lists):
    plt.plot(energy_list, label='{}'.format(order))
plt.legend()
plt.figure()

converged_energies = [np.sum([e - molecule.fci_energy for e in e_list]) for e_list in energy_lists]
categories = ['{}'.format(order) for order in orbital_orders]

index = np.argsort(converged_energies)


print(converged_energies)
print(categories)
plt.bar(np.array(categories)[index], np.array(converged_energies)[index])
plt.xticks(rotation=60)
plt.show()

#print(result.keys())
#from vqemulti.utils import store_wave_function, load_wave_function
#store_wave_function(result['coefficients'], result['ansatz'], filename='wf_h4.yml')
