from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
import numpy as np
import time


import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 1


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

# from vqemulti.utils import generate_reduced_hamiltonian, get_uccsd_operators
# print('len UCC: ', len(get_uccsd_operators(4, 4).terms))


benzene = get_molecule('../benzene.xyz')


n_compress_list = []
n_uncompress_list = []
cas_energy_list = []
n_active_list = []

range_pairs = range(1, 7)
for n_pair in range_pairs:
    #n_evol = 2

    n_orbitals_total = 21 + n_pair
    frozen_orbitals = 21 - n_pair

    # run classical calculation
    molecule = run_pyscf(benzene, run_fci=False, nat_orb=True, guess_mix=False, verbose=True,
                         frozen_core=frozen_orbitals, n_orbitals=n_orbitals_total)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons - frozen_orbitals * 2
    n_orbitals_active = n_orbitals_total - frozen_orbitals  # molecule.n_orbitals
    n_qubits = n_orbitals_active * 2
    hamiltonian = molecule.get_molecular_hamiltonian()

    from vqemulti.utils import store_hamiltonian
    #store_hamiltonian(hamiltonian, file='hamiltonian_extrap.npz')

    hamiltonian = get_fermion_operator(hamiltonian)
    print('H terms (not compress)', len(hamiltonian.terms))
    n_uncompress_list.append(len(hamiltonian.terms))

    hamiltonian.compress(1e-2)
    print('H terms (compress)', len(hamiltonian.terms))
    n_compress_list.append(len(hamiltonian.terms))

    #matrix = get_sparse_operator(hamiltonian).toarray()
    #print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])


    print("CASCI energy:", molecule.casci_energy)

    cas_energy_list.append(molecule.casci_energy)
    n_active_list.append(n_orbitals_active)



    matrix = get_sparse_operator(hamiltonian).toarray()
    print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])


    print('N electrons', n_electrons)
    print('N Orbitals', n_orbitals_active)
    # Choose specific pool of operators for adapt-VQE
    pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals_active)


    sym_orbitals = symmetrize_molecular_orbitals(molecule, 'D6h', skip=True, frozen_core=19, n_orbitals=23)
    pool = get_symmetry_reduced_pool(pool, sym_orbitals, threshold=0.5)  # using symmetry
    pool = pool.get_quibits_list(normalize=True)

    print('len pool', len(pool))


    from vqemulti.symmetry import get_pauli_symmetry_reduced_pool
    pool = get_pauli_symmetry_reduced_pool(pool)

    #from vqemulti.pool.tools import OperatorList
    #pool = OperatorList([pool[i] for i in range(0, len(pool), 4)])

    print('len pool', len(pool))


    print('\nADAPT_VQE POOL\n---------------------------')
    pool.print_compact_representation()

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

import matplotlib.pyplot as plt

for i in range(len(n_uncompress_list)):
    print('{:4} {:4} {:4} {:10.6f}'.format(n_active_list[i], n_uncompress_list[i], n_compress_list[i], cas_energy_list[i]))

plt.plot(n_active_list, n_uncompress_list, label='uncompress')
plt.plot(n_active_list, n_compress_list, label='compress')
plt.legend()
plt.xlabel('N active orbitals')
plt.ylabel('Hamiltonian terms')

plt.figure()
plt.plot(n_uncompress_list, cas_energy_list, label='uncompress')
plt.plot(n_compress_list, cas_energy_list, label='compress')
plt.legend()
plt.xlabel('Hamiltonian terms')
plt.ylabel('CAS energy [Ha]')



plt.figure()
plt.plot(n_active_list, cas_energy_list)
plt.legend()
plt.xlabel('N active orbitals')
plt.ylabel('CAS energy [Ha]')

plt.show()