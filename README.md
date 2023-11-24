VQEmulti
========

A variational quantum eigensolver (VQE) implementation based in Openfermion. This code
provides a common interface for different flavors of VQE. It uses cirq and pennylane
libraries for the energy and gradient simulations. 


VQE algorithms currently implemented
------------------------------------
- regular VQE
- adaptVQE


Installation
------------
```bash
cd vqemulti
pip -e install .
```


Basic example for regular VQE
-----------------------------
```python
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from vqemulti import vqe

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
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

# Get reference Hartree Fock state in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
print('hf reference', hf_reference_fock)

# Define Pennylane simulator
from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

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

print('Num operators: ', len(result['ansatz']))
print('Ansatz:\n', result['ansatz'])
print('Coefficients:\n', result['coefficients'])
```

Basic example for adaptVQE
----------------------------
```python
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.utils import generate_reduced_hamiltonian


# molecule definition
he2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                       ['He', [0, 0, 1.0]]],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='He2')

# run classical calculation
molecule = run_pyscf(he2_molecule)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# Choose specific pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons,
                                     n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

# define the simulator
simulator = Simulator(trotter=False,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

# run adaptVQE
result = adaptVQE(hamiltonian,
                  operators_pool,
                  hf_reference_fock,
                  opt_qubits=False,
                  threshold=0.002,
                  energy_simulator=simulator,
                  gradient_simulator=simulator,
                  )

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result['energy'])
print('Energy FullCI: ', molecule.fci_energy)

error = result['energy'] - molecule.fci_energy
print('Error:', error)

print('Ansatz:', result['ansatz'])
print('Coefficients:', result['coefficients'])
print('Operator Indices:', result['indices'])
print('Num operators: {}'.format(len(result['ansatz'])))
```