# example of oxygen molecule using frozen core of 7 orbitals (14 electrons)
# 1 occupied orbital (8) and 1 virtual orbital (9).
# calculation done in 4 qubits (8a, 8b, 9a, 9b).
# https://doi.org/10.48550/arXiv.2009.01872

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti import vqe

o2_molecule = MolecularData(geometry=[['O', [0, 0, 0]],
                                      ['O', [0, 0, 1.0]]],
                            # basis='3-21g',
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='O2')

# run classical calculation
molecule = run_pyscf(o2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 9  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=7)
# print(hamiltonian)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=7)

# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=7)
print('hf reference', hf_reference_fock)


print('Initialize VQE')
result = vqe(hamiltonian,
             uccsd_ansatz,
             hf_reference_fock)

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

print('Num operators: ', len(result['ansatz']))
print('Ansatz (compact representation):')
result['ansatz'].print_compact_representation()
print('Coefficients:\n', result['coefficients'])
