# Example of computing the density matrix of an adaptVQE wave function
# to compare with FullCI density matrix and get a measure of fidelity
# using the ansatz (list of operators) and coefficients

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd, get_pool_spin_complement_gsd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
import numpy as np


r = 3.0
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, r]],
                                      ['H', [0, 0, 2*r]],
                                      ['H', [0, 0, 3*r]]
                                      ],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

Configuration().mapping = 'jw'  # bk

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True, nat_orb=True, guess_mix=False)

# get FullCI density matrix in natural orbitals basis (for fidelity calculation)
nat_orb = h2_molecule.canonical_orbitals
can_orb = molecule._pyscf_data['scf'].mo_coeff

overlap_matrix = molecule._pyscf_data['scf'].get_ovlp(molecule._pyscf_data['mol'])

# dot_can = np.dot(can_orb.T, overlap_matrix @ can_orb)
# dot_nat = np.dot(nat_orb.T, overlap_matrix @ nat_orb)
# print('Check normalization: ', np.trace(dot_can), np.trace(dot_nat))

#  nat x can
trans_mat = np.dot(nat_orb.T, overlap_matrix @ can_orb)
# trans_mat = molecule.canonical_natural_trans_mat


print('Check trans_mat is unitary matrix', np.trace(trans_mat.T @ trans_mat))

#  (nat x nat) =  (nat x can)  x  (can x can) x  (can x nat)
nat_rdm_fci = trans_mat @ molecule.fci_one_rdm @ trans_mat.T
print('Check trace invariance under unit transformations: ', np.trace(molecule.fci_one_rdm), np.trace(nat_rdm_fci))


# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# build initial ansatz
ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

# Simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      shots=10000)

# FIRST CALCULATION
result_first = adaptVQE(hamiltonian,
                        operators_pool,
                        ansatz,
                        reference_dm=nat_rdm_fci,
                        # energy_simulator=simulator,
                        # gradient_simulator=simulator
                        )


print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result_first['energy'])
print('Energy FullCI: ', molecule.fci_energy)
print('Coefficients:', result_first['coefficients'])

print('\nFullCI density matrix')
print(np.round(nat_rdm_fci, decimals=6))
print(np.trace(nat_rdm_fci))


density_matrix = get_density_matrix(ansatz)

print('\nadaptVQE density matrix')
print(density_matrix)
print(np.trace(density_matrix))

# WF quantum fidelity (1: perfect match, 0: totally different)
fidelity = density_fidelity(density_matrix, nat_rdm_fci)
print('\nFidelity measure: {:7.4f}'.format(fidelity))
