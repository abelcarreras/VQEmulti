# example of He2 calculation using adaptVQE method
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti import NotConvergedError
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

# molecule definition
h6_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                       ['H', [0, 0, 3.0]],
                                      ['H', [0, 0, 6.0]],
                                      ['H', [0, 0, 9.0]],
                                      ['H', [0, 0, 12.0]],
                                      ['H', [0, 0, 15.0]]],
                             basis='sto-3g',
                             # basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='H6')

# run classical calculation
molecule = run_pyscf(h6_molecule, run_fci=False, nat_orb=False, guess_mix=False)

# get additional info about electronic structure properties
# from vqemulti.analysis import get_info
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons,
                           n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.method.adapt_vanila import AdapVanilla

method = AdapVanilla(gradient_threshold=1e-6,
                           diff_threshold=0,
                           coeff_tolerance=1e-10,
                           gradient_simulator=None,
                           prune = False,
                          )

try:
    # run adaptVQE
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      max_iterations=150  # maximum number of interations
                      )

except NotConvergedError as e:
    result = e.results


print('Final results\n-----------')

print('HF energy:', molecule.hf_energy)
print('Final adaptVQE energy:', result["energy"])
print('FullCI energy: {:.8f}'.format(molecule.fci_energy))

# Compare error vs FullCI calculation
error = result['energy'] - molecule.fci_energy
print('Error:', error)

# run results
print('Ansatz:', result['ansatz'])
print('Indices:', result['indices'])
print('Coefficients:', result['coefficients'])
print('Num operators: {}'.format(len(result['ansatz'])))
