from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti import NotConvergedError
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.method.adapt_vanila import AdapVanilla
from copy import deepcopy
import math
from pyscf import mcscf
import numpy as np
from openfermion import get_sparse_operator


# molecule definition
basis = '3-21g'
distance = 3
def water(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['O', [0, 0, 0]],
                                  ['H', [0, distance * math.sin(0.91193453416), distance * math.cos(0.91193453416) ]],
                                  ['H', [0, -distance * math.sin(0.91193453416), distance * math.cos(0.91193453416)]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H2O')
    return mol


# molecule definition
h2o_molecule = water(3, basis=basis)

print(h2o_molecule.geometry)

# run classical calculation
molecule = run_pyscf(h2o_molecule, run_fci=False, nat_orb=False, guess_mix=False, frozen_core=1, verbose=True, n_orbitals=9)
print('casci_energy: ', molecule.casci_energy)
print('n_orbitals: ', molecule.n_orbitals)
print()

from openfermion import get_fermion_operator
print(len(get_fermion_operator(molecule.get_molecular_hamiltonian()).terms))
#matrix = get_sparse_operator(molecule.get_molecular_hamiltonian()).toarray()
#print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])

molecule = run_pyscf(h2o_molecule, run_fci=False, nat_orb=False, guess_mix=False)
print(molecule.fci_energy)

# Assuming run_pyscf returns a PyscfMolecularData object
# Try:
scf_obj = molecule._pyscf_data['scf']  # This is the PySCF RHF object

# Number of active orbitals
ncas = 8

# Calculate the number of active electrons
# Freeze 1 orbital => freeze 2 electrons (closed-shell)
nelecas = molecule.n_electrons - 2
print('nelecas: ', nelecas)

# Perform CASCI
mc = mcscf.CASCI(scf_obj, ncas, nelecas)
mc.frozen = [0]

# Run CASCI
casci_result = mc.kernel()[0]

print('casci_result: ',  casci_result)


# get additional info about electronic structure properties
# from vqemulti.analysis import get_info
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 9 # for 3-21g
#n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=1)
print(len(get_fermion_operator(hamiltonian).terms))


#matrix = get_sparse_operator(hamiltonian).toarray()
#print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])



# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons,
                           n_orbitals=n_orbitals, frozen_core=1)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=1)
print(hf_reference_fock)