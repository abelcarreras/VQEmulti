# example of He2 calculation using adaptVQE method
from utils import get_hf_reference_in_fock_space
from pool import get_pool_singlet_sd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from adapt_vqe import adaptVQE
import matplotlib.pyplot as plt
from utils import generate_reduced_hamiltonian
from errors import NotConvergedError

# molecule definition
h2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                      ['He', [0, 0, 1.0]]],
                            basis='3-21g',
                            # basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='He2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True)

# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons,
                           n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)


try:
    # run adaptVQE
    result = adaptVQE(pool,
                      hamiltonian,
                      hf_reference_fock,
                      opt_qubits=False,
                      threshold=0.01,
                      max_iterations=15
                      )

except NotConvergedError as e:
    result = e.results


print('Final results\n-----------')

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


plt.plot(result['iterations']['energies'])
plt.title('Energy')
plt.ylabel('Hartree')

plt.figure()

plt.plot(result['iterations']['norms'])
plt.title('Gradient')
plt.ylabel('Hartree')
plt.show()