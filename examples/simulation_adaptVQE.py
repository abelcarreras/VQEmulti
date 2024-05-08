# example of He2 calculation using adaptVQE method
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.optimizers import OptimizerParams
from vqemulti import NotConvergedError
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt


# general configuration
Configuration().verbose = True

# molecule definition
he2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                       ['He', [0, 0, 1.0]]],
                             basis='6-31g',
                             multiplicity=1,
                             charge=0,
                             description='He2')

# run classical calculation
molecule = run_pyscf(he2_molecule, run_fci=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# define simulator
simulator = Simulator(trotter=False, test_only=True, hamiltonian_grouping=True, shots=250)

# define optimizer (check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': 1e-2})
opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

try:
    # run adaptVQE
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      energy_simulator=simulator,
                      gradient_simulator=simulator,
                      energy_threshold=1e-2,
                      opt_qubits=False,  # use fermion operators
                      max_iterations=15,  # maximum number of interations
                      optimizer_params=opt_cobyla # optimizer parameters
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

# plot results
plt.plot(result['iterations']['energies'])
plt.title('Energy')
plt.ylabel('Hartree')

plt.figure()

plt.plot(result['iterations']['norms'])
plt.title('Gradient')
plt.ylabel('Hartree')
plt.show()
