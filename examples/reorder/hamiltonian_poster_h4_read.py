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
from vqemulti.utils import store_wave_function, load_wave_function, store_hamiltonian, load_hamiltonian
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_adapt_vqe_energy
from vqemulti.energy import get_adapt_vqe_energy

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 1
config.mapping = 'pc'

print('mapping: ', config.mapping)

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
molecule = run_pyscf(h4, run_fci=True, nat_orb=False, guess_mix=False, verbose=True, frozen_core=0)
print('FullCI energy: : ', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()

# store_hamiltonian(hamiltonian, file='hamiltonian_bk.npz')
# hamiltonian = load_hamiltonian('hamiltonian.npz')


# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)


print("HF energy:", molecule.hf_energy)

orbital_orders = []
energy_list = []
energy_error_list = []
cnot_list = []
depth_ave_list = []
depth_max_list = []

for n_oper in range(10):
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True,
                          shots=1024,
                          )
    # simulator = None

    coefficients, ansatz = load_wave_function('wf_h4_f{}.yml'.format(n_oper), qubit_op=False)
    # coefficients = coefficients[:0]
    # ansatz = ansatz[:0]
    # print(coefficients, ansatz)

    energy = get_adapt_vqe_energy(coefficients,ansatz, hf_reference_fock, hamiltonian, simulator)

    print('Final adaptVQE energy:', energy)
    print("FCI energy:", molecule.fci_energy)
    print("Error (respect to FCI): {:10.3e}".format(energy - molecule.fci_energy))

    print('simulation energy')
    simulator.print_statistics()
    # print(simulator._circuit_gates)

    energy_list.append(energy)
    energy_error_list.append(energy - molecule.fci_energy)

    cnot_list.append(simulator._circuit_gates['CNOT'])
    depth_ave_list.append(np.average(simulator._circuit_count))
    depth_max_list.append(np.max(simulator._circuit_count))

plt.plot(energy_list)
plt.legend()
plt.show()

for i in range(10):
    print('{:10.6f} {:10.6f}  {:10}  {:10}'.format(energy_list[i],
                                                   energy_error_list[i],
                                                   cnot_list[i],
                                                   depth_max_list[i]))
