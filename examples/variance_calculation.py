# example to compute the variance of a sampled (using simulator) VQE calculation
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.energy.simulation import simulate_vqe_energy_square, simulate_vqe_energy
from vqemulti.energy import exact_vqe_energy
import numpy as np


# define molecule
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 1.0]]],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2'
                             )

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 2  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=0)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=0)

# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=0)
print('hf reference', hf_reference_fock)

# Compute VQE to get a wave function (coefficients/ansatz)  [no simulation]
print('Initialize VQE')
result = vqe(hamiltonian, uccsd_ansatz, hf_reference_fock)

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
print('------------------------')

# Configuration().verbose = 2
# perform an energy evaluation for the calculated wave function (coefficients/ansatz)
exact = exact_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian)
print('Exact energy: ', exact)

# compute the variance as Var = E[X^2] - E[X]^2 with simulator
n_measures = 100
simulator = Simulator(trotter=True, test_only=False, shots=10000)

e = np.average([simulate_vqe_energy(result['coefficients'],
                                    result['ansatz'],
                                    hf_reference_fock, hamiltonian, simulator)
                for _ in range(n_measures)])

e2 = np.average([simulate_vqe_energy_square(result['coefficients'],
                                            result['ansatz'],
                                            hf_reference_fock, hamiltonian, simulator)
                 for _ in range(n_measures)])

print('E[X^2]: ', e2)
print('E[X]^2: ', e**2)
print('E[X]: ', e)

variance = e2 - e**2
print('Variance: ', variance)
std = np.sqrt(abs(variance) / n_measures)
print('standard error: ', std)
