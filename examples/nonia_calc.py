from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd, get_pool_spin_complement_gsd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.errors import NotConvergedError
from generate_mol import linear_h4_mol

# molecule definition
h2_molecule = linear_h4_mol(3, basis='3-21g')

# run classical calculation
molecule = run_pyscf(h2_molecule, nat_orb=False, run_fci=True, verbose=True)


'''
nat_orb = molecule.canonical_orbitals
can_orb = molecule._pyscf_data['scf'].mo_coeff

overlap_matrix =  molecule._pyscf_data['scf'].get_ovlp( molecule._pyscf_data['mol'])

dot_can = np.dot(can_orb.T, overlap_matrix @ can_orb)
dot_nat = np.dot(nat_orb.T, overlap_matrix @ nat_orb)

print('Check normalization: ', np.trace(dot_can), np.trace(dot_nat))

#  nat x can
trans_mat = np.dot(nat_orb.T, overlap_matrix @ can_orb)

print('Check trans_mat is unitary matrix', np.trace(trans_mat.T @ trans_mat))

nat_rdm_fci = trans_mat @ molecule.fci_one_rdm @ trans_mat.T

print('Check trace invariance under unit transformations: ', np.trace(molecule.fci_one_rdm), np.trace(nat_rdm_fci))
'''


# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = 4 # molecule.n_electrons
n_orbitals = 8  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)



# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons,
                            n_orbitals=n_orbitals)


# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator


simulator = Simulator(trotter=True,
                        trotter_steps=1,
                        test_only=True,
                        shots=1000)


# run adaptVQE
result = adaptVQE(hamiltonian,
                    pool,
                    hf_reference_fock,
                    opt_qubits=False,
                    max_iterations= 200,
                    coeff_tolerance=1e-4,
                    energy_threshold=1e-4,
                    #gradient_threshold=1e-9,
                    #energy_simulator= simulator,
                    #gradient_simulator=None,
                    reference_dm = molecule.fci_one_rdm,
                    #reference_dm = nat_rdm_fci
                    )

print(result)