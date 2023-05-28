VQEmulti
========

A variational quantum eigensolver (VQE) implementation based in Openfermion. This code
provides a common interface for different flavors of VQE. It uses cirq and pennylane
libraries for the energy and gradient simulations. 


VQE algorithms currently implemented
------------------------------------
- regular VQE
- adaptVQE


Basic example for regular VQE
-----------------------------
```python
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from utils import generate_reduced_hamiltonian, get_uccsd_operators, get_hf_reference_in_fock_space
from vqe import vqe

h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 1.0]]],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

# obtain hamiltonian in fermion operators
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz in fermion operators
uccsd_ansatz = get_uccsd_operators(n_electrons, n_orbitals, frozen_core=7)

# Get reference Hartree Fock state in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
print('hf reference', hf_reference_fock)

# Define Pennylane simulator
from simulators.penny_simulator import PennylaneSimulator as Simulator

simulator = Simulator(trotter=True,     # use Trotter transformation
                      trotter_steps=1,  # define Trotter transformation steps
                      shots=1000)       # define number of shots

print('Initialize VQE')
result = vqe(hamiltonian,                # hamiltonian in fermion operators 
             uccsd_ansatz,               # ansatz in fermion operators
             hf_reference_fock,          # reference vector in Fock space
             energy_simulator=simulator) # use simulator 

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

print('Num operators: ', len(result['operators']))
print('Operators:\n', result['operators'])
print('Coefficients:\n', result['coefficients'])
```

Basic example for adaptVQE
----------------------------
```python
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from pool_definitions import get_pool_singlet_sd
from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from adapt_vqe import adaptVQE


h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 0.74]]],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 2  # molecule.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of fermion operators for adapt-VQE
operators_pool = get_pool_singlet_sd(electronNumber=n_electrons,
                                     orbitalNumber=n_orbitals)

# Get reference Hartree Fock state in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Define Pennylane simulator
from simulators.penny_simulator import PennylaneSimulator as Simulator
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      shots=1000)

result, iterations = adaptVQE(operators_pool,               # fermionic operators
                              hamiltonian,                  # fermionic hamiltonian
                              hf_reference_fock,            # reference vector in Fock space
                              threshold=0.1,                # adaptVQE convergence gradient threshold 
                              energy_simulator=simulator,   # simulator for energy calculation
                              gradient_simulator=simulator) # simulator for gradient calculation

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result['energy'])
print('Energy FullCI: ', molecule.fci_energy)

error = result['energy'] - molecule.fci_energy
print('Error:', error)

print('Ansatz:', result['ansatz'])
print('Indices:', result['indices'])
print('Coefficients:', result['coefficients'])
print('Num operators: {}'.format(len(result['ansatz'])))
```